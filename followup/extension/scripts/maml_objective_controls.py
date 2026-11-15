#!/usr/bin/env python3
"""Pair zero-step and after-adaptation training on the exact released MAML task stream."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "delivery/submission_20261115/repository"
sys.path.insert(0, str(FROZEN / "scripts"))

import final_model_engine as base
import research_model_engine as engine
from load_research_inputs import load_inputs

ORIGINAL = ROOT / "results/final_study_20261114/mac_research/afml_research/models"
DEFAULT_OUT = ROOT / "results/explanatory_extension_20261115/maml_objective"
SEEDS = (17, 43, 71)
SOURCE_STAGES = (300, 1350)
SCHEDULES = ((0.0, 0), (0.01, 3), (0.03, 5), (0.03, 10))
PERIODS = {"validation": "secondary_2024", "test": "test_2025"}
SPEC = {"architecture": "manual_gru", "hidden": 64, "layers": 2}
ARMS = ("zero_query_objective", "after_five_query_objective")
BASE_SEED = 7919


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_config(path):
    return json.loads(path.read_text())


def checkpoint(kind, seed):
    name = "maml" if kind == "maml" else "manual_gru"
    return ORIGINAL / "source_states" / f"{name}_s{seed}.pt"


def journal(seed):
    return ORIGINAL / "meta_episodes" / f"refit_meta_maml_s{seed}.csv.gz"


def protocol(out):
    inputs = [Path(__file__),
              *[FROZEN / "scripts" / f"{module}.py" for module in
                ("final_model_engine", "research_model_engine", "load_research_inputs")],
              FROZEN / "data/frozen/INPUT_MANIFEST.json",
              ORIGINAL / "effective_config.json", ORIGINAL / "source_selections.json",
              *[checkpoint(kind, seed) for seed in SEEDS for kind in ("maml", "pooled")],
              *[journal(seed) for seed in SEEDS],
              *[ORIGINAL / "checkpoints" / f"refit_meta_maml_s{seed}.pt" for seed in SEEDS]]
    spec = {
        "stage": "post hoc on previously inspected target years; no target selection",
        "hypothesis": (
            "An objective penalizing forecasts after local updates can yield a worse "
            "zero-step initialization than directly minimizing zero-step query errors."
        ),
        "contrast": (
            "Same initial weights, four task-draws/update, donor data, seed, purged "
            "episode labels, sampled 32 query examples, query residual MSE, Adam outer "
            "optimization, clipping and update count. Only the training objective "
            "changes: evaluate that query MSE at zero or after five support SGD steps."
        ),
        "donor_batches": (
            "Reconstruct archived donor episode and minibatch RNG streams. Validate "
            "every label-set hash and archived RNG state. Evaluate the untouched query "
            "objective after either zero or five support steps."
        ),
        "support_sgd": "Rate .03, five steps, 32 sampled rows/step, grad norm clip 1.",
        "stages": list(SOURCE_STAGES),
        "methods": [*ARMS, "released_pooled", "released_maml"],
        "seeds": list(SEEDS),
        "budgets": [30, 90, 180],
        "evaluation": (
            "All frozen original 2024/2025 target masks and horizon labels. Fit "
            "target SGD on the frozen local support, same batches and schedule between "
            "objectives, no target selection. Compare seed/city/budget/horizon means "
            "at zero, .01x3, .03x5 and .03x10. Report all schedules."
        ),
        "evaluation_schedules": [list(s) for s in SCHEDULES],
        "interpretation": (
            "Isolates the after-update objective within the original episodic data "
            "distribution. Original pooled training uses a different sampling and "
            "optimization recipe; comparisons with it do not isolate the objective."
        ),
        "evaluation_donor_measure": (
            "Seven common 2023 label dates per donor, selected by the released "
            "checkpoint investigation hash rule; descriptive training-retention "
            "measure only, not held-out donor evidence."
        ),
        "device": "CPU for both arms, one thread per concurrent seed",
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "PROTOCOL.json"
    if path.exists():
        assert json_config(path) == spec, "Frozen protocol or input changed"
    else:
        base._atomic_json(path, spec)
    return digest(path)


def model(seed):
    return engine._new_model(SPEC, seed, torch.device("cpu"))


def validate_journal(episode, si, qi, recorded, step, task, dates):
    assert recorded.step == step and recorded.task == task and recorded.algorithm == "maml"
    for key, value in episode.items():
        assert getattr(recorded, key) == value, (step, task, key)
    for name, indices in (("support", si), ("query", qi)):
        labels = sorted(set(str(value)[:10] for value in dates[indices]))
        assert engine._dates_digest(labels) == getattr(recorded, f"{name}_label_dates_sha256")
    assert recorded.inner_steps == 5 and recorded.inner_lr == .03
    assert recorded.support_batch_examples == recorded.query_batch_examples == 32


def get_donor_panel(train):
    batches = []
    for city, rows in train.loc[train.date.dt.year == 2023].groupby("city"):
        counts = rows.groupby("date").horizon.nunique()
        dates = counts[counts == 3].index.tolist()
        selected = sorted(dates, key=lambda date: hashlib.sha256(
            f"{city}|{date:%Y-%m-%d}".encode()).hexdigest())[:7]
        batches.append(rows[rows.date.isin(selected)])
    panel = pd.concat(batches).reset_index(drop=True)
    assert panel.city.nunique() == 23 and len(panel) == 23 * 7 * 3
    return panel


def summarize_scores(raw, seed, source_stage, arm, rate, steps, budget):
    key = ["period", "city", "horizon"]
    error = raw.y_pred - raw.y_true
    metrics = raw.assign(absolute_error=error.abs(), squared_error=error**2, error=error)
    scores = metrics.groupby(key, as_index=False).agg(
        n=("y_pred", "size"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"),
        bias=("error", "mean"))
    scores["rmse"] = np.sqrt(scores.mse)
    return scores.assign(seed=seed, source_stage=source_stage, objective=arm,
                         rate=rate, steps=steps, budget=budget)


def evaluate(models, seed, stage, data, config, out):
    device = torch.device("cpu")
    frames = []
    score_rows = []
    donor_panel = get_donor_panel(data["donor_full"])
    for arm, fitted in models.items():
        donor = base._prediction_rows(donor_panel, base.predict_neural(fitted, donor_panel, device),
                                      method=arm, budget=0, seed=seed, period="seen_donor_retention_2023")
        score_rows.append(summarize_scores(donor, seed, stage, arm, 0.0, 0, 0))
        for city, parts in sorted(data["targets"].items()):
            zero = {key: base.predict_neural(fitted, parts[key], device) for key in PERIODS}
            for budget in config["budgets"]:
                support = base.select_budget(parts["adapt"], budget)
                assert support.date.nunique() == budget and len(support) == 3 * budget
                assert support.date.max() <= pd.Timestamp("2023-12-31")
                for rate, steps in SCHEDULES:
                    if steps:
                        adapted = engine.fit_supervised(copy.deepcopy(fitted), support,
                            {"lr": rate, "steps": steps, "optimizer": "sgd", "weight_decay": 0.0},
                            seed=seed, c=config, device=device, adaptation=True).model
                        forecasts = {key: base.predict_neural(adapted, parts[key], device) for key in PERIODS}
                    else:
                        forecasts = zero
                    for local_key, period in PERIODS.items():
                        raw = base._prediction_rows(parts[local_key], forecasts[local_key],
                            method=arm, budget=budget, seed=seed, period=period)
                        assert (pd.to_datetime(raw.issue_date) < pd.to_datetime(raw.date)).all()
                        score_rows.append(summarize_scores(raw, seed, stage, arm, rate, steps, budget))
                        if steps in (0, 5):
                            frames.append(raw.assign(source_stage=stage, rate=rate, steps=steps))
    scores = pd.concat(score_rows, ignore_index=True)
    forecasts = pd.concat(frames, ignore_index=True)
    assert np.isfinite(scores[["mae", "rmse"]]).all().all()
    base._atomic_csv(out / f"metrics_s{seed}_step{stage}.csv", scores)
    base._atomic_csv(out / f"forecasts_s{seed}_step{stage}.csv.gz", forecasts)


def run(seed, out):
    started = time.monotonic()
    protocol_hash = protocol(out)
    config = json_config(ORIGINAL / "effective_config.json")
    config["_deadline"] = float("inf")
    preset = json_config(ORIGINAL / "source_selections.json")["meta"]["maml"]
    hyper = preset["preset"]
    assert (hyper["inner_steps"], hyper["inner_lr"], hyper["outer_lr"], preset["steps"]) == (5, .03, .0003, 1350)
    data = load_inputs(FROZEN / "data/frozen", out / f"input_s{seed}")[0]
    train = data["donor_full"].reset_index(drop=True)
    assert train.city.nunique() == 23 and len(train) == 71829
    assert train.date.max() <= pd.Timestamp("2023-12-31") and set(train.city).isdisjoint(data["targets"])
    device = torch.device("cpu")
    arrays = base._arrays(train, device)
    expected = list(pd.read_csv(journal(seed)).itertuples(index=False))
    assert len(expected) == 1350 * config["meta_tasks"]
    models = {arm: model(seed) for arm in ARMS}
    assert all(torch.equal(left, right) for left, right in zip(
        models[ARMS[0]].parameters(), models[ARMS[1]].parameters()))
    opts = {arm: torch.optim.Adam(network.parameters(), lr=hyper["outer_lr"]) for arm, network in models.items()}
    references = {"released_pooled": model(seed), "released_maml": model(seed)}
    for name, net in references.items():
        net.load_state_dict(base._torch_load(checkpoint(name.removeprefix("released_"), seed), device)["state_dict"])
    sampler = base.EpisodeSampler(train, config["budgets"], config["meta_query_days"],
                                  config["purge_days"], seed + 177)
    rng = np.random.default_rng(seed + BASE_SEED)
    source_loss, parity = [], []
    for step in range(1, preset["steps"] + 1):
        sample_batch = []
        for task in range(config["meta_tasks"]):
            si, qi, episode = sampler.sample()
            support_batches = [rng.choice(si, size=min(config["meta_batch_size"], len(si)), replace=True)
                               for _ in range(hyper["inner_steps"])]
            query = rng.choice(qi, size=min(config["meta_batch_size"], len(qi)), replace=False)
            validate_journal(episode, si, qi, expected[(step-1)*config["meta_tasks"]+task], step, task, arrays.dates)
            assert set(query).issubset(qi) and all(set(batch).issubset(si) for batch in support_batches)
            sample_batch.append((support_batches, query))
        for arm, network in models.items():
            network.train()
            totals = [torch.zeros_like(parameter) for parameter in network.parameters()]
            losses = []
            for support_batches, query in sample_batch:
                inner_batches = support_batches if arm == ARMS[1] else []
                gradients, loss = engine._meta_gradient(network, arrays.x, arrays.residual,
                    inner_batches, query, hyper, "maml", float(config["grad_clip"]))
                losses.append(loss)
                for total, gradient in zip(totals, gradients):
                    total.add_(gradient)
            opts[arm].zero_grad(set_to_none=True)
            for parameter, total in zip(network.parameters(), totals):
                parameter.grad = total / config["meta_tasks"]
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(config["grad_clip"]))
            opts[arm].step()
            source_loss.append({"step": step, "objective": arm, "query_mse_at_objective_state": np.mean(losses)})
        if step % 50 == 0:
            print(f"seed={seed} source_step={step} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if step in SOURCE_STAGES:
            saved = {arm: base._state(net) for arm, net in models.items()}
            base._torch_save(out / f"trained_s{seed}_step{step}.pt", saved)
            evaluate({**models, **references}, seed, step, data, config, out)
            print(f"seed={seed} stage={step} evaluated elapsed={time.monotonic()-started:.1f}s", flush=True)
    released = base._torch_load(ORIGINAL / "checkpoints" / f"refit_meta_maml_s{seed}.pt", device)
    assert rng.bit_generator.state == released["inner_rng"]
    assert sampler.rng.bit_generator.state == released["sampler_rng"]
    for city, parts in data["targets"].items():
        raw = parts["test"]
        diff = abs(base.predict_neural(references["released_maml"], raw, device)
                   -base.predict_neural(models[ARMS[1]], raw, device))
        parity.append({"city": city, "mean_abs_aqi": float(np.mean(diff)), "max_abs_aqi": float(np.max(diff))})
    base._atomic_csv(out / f"source_training_s{seed}.csv", pd.DataFrame(source_loss))
    base._atomic_json(out / f"receipt_s{seed}.json", {
        "protocol_sha256": protocol_hash, "seed": seed, "episodes_exact": len(expected),
        "archived_final_rngs_exact": True, "retained_source_stages": SOURCE_STAGES,
        "cpu_replay_vs_released_mps_maml_prediction_difference": parity,
        "elapsed_seconds": time.monotonic()-started, "platform": platform.platform(),
        "torch": torch.__version__,
        "sha256": {p.name: digest(p) for p in out.glob(f"*_s{seed}*") if p.is_file() and p.suffix != ".log"},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.action == "freeze":
        print(protocol(args.out))
    elif args.seed is None:
        parser.error("--seed required for run")
    else:
        run(args.seed, args.out)


if __name__ == "__main__":
    main()
