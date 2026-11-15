#!/usr/bin/env python3
"""Balanced-volume donor-panel sensitivity using the released AFML training engine."""
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
COUNTS = (6, 12, 23)
CHAINS = {"a": 17, "b": 43, "c": 71}
YEAR_COUNTS = {2020: 552, 2021: 828, 2022: 1104, 2023: 828}
TARGET_BUDGETS = (30, 90, 180)
SOURCE_STEPS = 1200
META_STEPS = 300
SPEC = {"architecture": "manual_gru", "hidden": 64, "layers": 2}
SCHEDULES = {"sgd01x3": (.01, 3), "sgd03x10": (.03, 10)}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design(args) -> dict:
    return {
        "status": "Post hoc on the already inspected original 2024 and 2025 results",
        "donor_counts": COUNTS,
        "chain_training_seeds": CHAINS,
        "source_city_label_dates_by_year": YEAR_COUNTS,
        "source_row_selection": "SHA256-ranked nested donor prefixes per chain and SHA256-ranked city-date prefixes with exact per-city yearly quotas; every selected label has all three horizons",
        "source_model": SPEC,
        "supervised": {"lr": .001, "weight_decay": .0001, "optimizer": "adam", "steps": SOURCE_STEPS, "batch": 256},
        "maml": {"inner_lr": .03, "inner_steps": 5, "outer_lr": .0003, "steps": META_STEPS,
                 "task_count": 4, "support_labels": 90, "query_labels": 14, "purge_intervening_days": 9, "batch": 32},
        "target_budgets": TARGET_BUDGETS,
        "target_schedules": {"zero": (0, 0), **SCHEDULES},
        "target_evaluation": "Released chronological support selector; original common horizon rows in 2024 and 2025; every condition scored; no target optimization or arm selection",
        "interpretation": "Equal city-date volume and within-method update counts. City identities and per-city/episode temporal density vary with count. A fixed seed is paired with each independent panel chain; variability cannot separate composition from initialization. MAML uses 300 outer updates / fixed 90-label training tasks for bounded compute, versus 1350 / mixed 30/90/180-label tasks in original refits. No convergence claim.",
        "source_sha256": {
            "experiment": file_sha(Path(__file__)),
            "input_manifest": file_sha(args.data / "INPUT_MANIFEST.json"),
            **{name: file_sha(args.engine / name) for name in
               ("final_model_engine.py", "research_model_engine.py", "load_research_inputs.py")},
        },
    }


def panel(source: pd.DataFrame, chain: str, count: int):
    complete = source.groupby(["city", "date"], sort=True).horizon.agg(["size", "nunique"])
    valid = complete[(complete["size"] == 3) & (complete["nunique"] == 3)].reset_index()[["city", "date"]]
    valid["year"] = valid.date.dt.year
    roster = sorted(source.city.unique(), key=lambda city: sha(f"matched-diversity-v1|{chain}|city|{city}"))[:count]
    keys = []
    for city in roster:
        for year, total in YEAR_COUNTS.items():
            assert total % count == 0
            candidates = valid[(valid.city == city) & (valid.year == year)].date
            ranked = sorted(candidates, key=lambda date: sha(f"matched-diversity-v1|{chain}|{city}|{date:%Y-%m-%d}"))
            quota = total // count
            if len(ranked) < quota:
                raise ValueError(f"Insufficient complete dates for {chain}/{count}/{city}/{year}: {len(ranked)} < {quota}")
            keys.extend((city, date) for date in ranked[:quota])
    labels = pd.DataFrame(keys, columns=["city", "date"]).sort_values(["city", "date"]).reset_index(drop=True)
    train = source.merge(labels, on=["city", "date"], how="inner", validate="many_to_one")
    train = train.sort_values(["city", "date", "horizon"]).reset_index(drop=True)
    assert len(labels) == sum(YEAR_COUNTS.values()) and len(train) == 3 * len(labels)
    assert train.issue_date.ge("2020-01-01").all() and train.date.le("2023-12-31").all()
    return roster, labels, train


def evaluate(model, targets, *, chain, count, seed, method, config, device, base, engine):
    for city, parts in targets.items():
        query = pd.concat([parts["validation"].assign(period="secondary_2024"),
                           parts["test"].assign(period="test_2025")], ignore_index=True)
        identity = ["city", "station_id", "date", "issue_date", "period", "horizon", "actual"]
        zero = base.predict_neural(model, query, device)
        query_predictions = query.rename(columns={"y": "actual"})[identity].copy()
        query_predictions["zero"] = zero
        for budget in TARGET_BUDGETS:
            support = base.select_budget(parts["adapt"], budget)
            assert support.date.nunique() == budget and len(support) == 3 * budget
            current = query_predictions.assign(budget=budget)
            for name, (lr, steps) in SCHEDULES.items():
                settings = {"optimizer": "sgd", "lr": lr, "steps": steps, "weight_decay": 0.0}
                fitted = engine.fit_supervised(copy.deepcopy(model), support, settings,
                                               seed=seed, c=config, device=device, adaptation=True)
                assert fitted.actual_steps == steps
                current[name] = base.predict_neural(fitted.model, query, device)
            yield current.assign(chain=chain, donor_count=count, seed=seed, method=method)


def run(args, base, engine, data):
    chain = args.chain
    seed = CHAINS[chain]
    block = args.output / f"chain_{chain}"
    block.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = engine.effective_config({"budgets": [90]})
    for count in COUNTS:
        roster, labels, train = panel(data["donor_full"], chain, count)
        sampler = base.EpisodeSampler(train, [90], 14, 9, seed)
        starts = {city: len(value["starts"]) for city, value in sampler.groups.items()}
        base._atomic_csv(block / f"labels_{count}.csv", labels)
        base._atomic_json(block / f"panel_{count}.json", {"donors": roster, "eligible_episode_starts_by_city": starts,
             "unique_calendar_dates": int(labels.date.nunique()),
             "per_city_first_last": {city: [str(group.date.min())[:10], str(group.date.max())[:10]]
                                      for city, group in labels.groupby("city")},
             "rows": len(train), "label_dates": len(labels)})
        for method in ("pooled", "maml"):
            name = f"{method}_{count}"
            done = block / f"{name}.json"
            if done.exists():
                prior = json.loads(done.read_text())
                assert prior["prediction_sha256"] == file_sha(block / f"predictions_{name}.csv.gz")
                assert prior["checkpoint_sha256"] == file_sha(block / f"{name}.pt")
                continue
            print(f"train {chain} {count} {method}", flush=True)
            began = time.perf_counter()
            ckpt = block / f"{name}.pt"
            if method == "pooled":
                preset = {"lr": .001, "weight_decay": .0001, "optimizer": "adam", "steps": SOURCE_STEPS}
                fitted = engine.fit_supervised(engine._new_model(SPEC, seed, device), train, preset,
                                               seed=seed, c=config, device=device, checkpoint=ckpt)
            else:
                preset = {"inner_lr": .03, "inner_steps": 5, "outer_lr": .0003, "steps": META_STEPS}
                fitted = engine.fit_meta("maml", SPEC, train, preset, seed=seed, c=config,
                                         device=device, checkpoint=ckpt)
                episode = pd.DataFrame(fitted.episodes)
                assert len(episode) == META_STEPS * 4 and episode.full_second_order.all()
                assert (episode.purge_intervening_days >= 9).all() and (episode.budget == 90).all()
                base._atomic_csv(block / f"episodes_{count}.csv.gz", episode)
            assert fitted.actual_steps == preset["steps"]
            prediction_seconds = time.perf_counter()
            forecasts = pd.concat(evaluate(fitted.model, data["targets"], chain=chain, count=count, seed=seed,
                                          method=method, config=config, device=device, base=base, engine=engine),
                                  ignore_index=True)
            path = block / f"predictions_{name}.csv.gz"
            base._atomic_csv(path, forecasts)
            base._atomic_json(done, {"chain": chain, "seed": seed, "donor_count": count, "method": method,
                "steps": fitted.actual_steps, "fit_seconds": prediction_seconds - began,
                "prediction_rows": len(forecasts), "prediction_sha256": file_sha(path),
                "checkpoint_sha256": file_sha(ckpt), "training_history": fitted.history,
                "device": args.device, "torch": torch.__version__, "threads": args.threads,
                "platform": platform.platform()})
            print(f"done {chain} {count} {method}: {time.perf_counter()-began:.2f} seconds", flush=True)


def summarize(args, base, data):
    outputs = []
    canonical = pd.concat([pd.concat([p["validation"].assign(period="secondary_2024"),
        p["test"].assign(period="test_2025")]) for p in data["targets"].values()])
    keys = ["city", "station_id", "date", "issue_date", "period", "horizon"]
    reference = canonical.set_index(keys).y.sort_index()
    for chain, seed in CHAINS.items():
        block = args.output / f"chain_{chain}"
        for count in COUNTS:
            for method in ("pooled", "maml"):
                name = f"{method}_{count}"
                file = block / f"predictions_{name}.csv.gz"
                receipt = json.loads((block / f"{name}.json").read_text())
                assert file_sha(file) == receipt["prediction_sha256"]
                forecasts = pd.read_csv(file, dtype={"station_id": str}, parse_dates=["date", "issue_date"])
                assert set(forecasts.budget) == set(TARGET_BUDGETS)
                assert set(forecasts.chain) == {chain} and set(forecasts.seed) == {seed}
                assert set(forecasts.method) == {method} and set(forecasts.donor_count) == {count}
                for _, prediction in forecasts.groupby("budget"):
                    actual = prediction.set_index(keys).actual.sort_index()
                    assert actual.index.equals(reference.index)
                    assert np.array_equal(actual.to_numpy(), reference.to_numpy())
                new = forecasts.melt(id_vars=["chain", "seed", "donor_count", "method", "budget", "city",
                    "horizon", "period", "actual"], value_vars=["zero", *SCHEDULES],
                    var_name="adaptation", value_name="predicted")
                error = new.predicted - new.actual
                assert np.isfinite(error).all()
                new["ae"], new["se"] = error.abs(), error.pow(2)
                metrics = new.groupby(["chain", "seed", "donor_count", "method", "budget",
                    "city", "horizon", "period", "adaptation"], as_index=False).agg(
                        n=("ae", "size"), mae=("ae", "mean"), mse=("se", "mean"))
                metrics["rmse"] = np.sqrt(metrics.pop("mse"))
                outputs.append(metrics)
    scores = pd.concat(outputs, ignore_index=True)
    assert len(scores) == 5832
    base._atomic_csv(args.output / "city_metrics.csv", scores)
    group = ["chain", "seed", "donor_count", "method", "budget", "horizon", "period", "adaptation"]
    macro = scores.groupby(group, as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), cities=("city", "size"))
    assert (macro.cities == 6).all()
    base._atomic_csv(args.output / "macro_metrics.csv", macro)
    paired = macro.pivot(index=[k for k in group if k != "donor_count"], columns="donor_count", values="mae").reset_index()
    paired["mae_12_minus_6"], paired["mae_23_minus_6"], paired["mae_23_minus_12"] = (
        paired[12] - paired[6], paired[23] - paired[6], paired[23] - paired[12])
    base._atomic_csv(args.output / "paired_donor_count_changes.csv", paired)
    base._atomic_json(args.output / "SUMMARY.json", {"arms": len(outputs), "city_metric_rows": len(scores),
        "macro_rows": len(macro), "paired_rows": len(paired), "all_target_masks_verified": True,
        "protocol_sha256": file_sha(args.output / "PROTOCOL.json")})


def main():
    frozen = ROOT / "delivery/submission_20261115/repository"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "check", "run", "summarize"])
    parser.add_argument("--chain", choices=CHAINS)
    parser.add_argument("--engine", type=Path, default=frozen / "scripts")
    parser.add_argument("--data", type=Path, default=frozen / "data/frozen")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/explanatory_investigation_20261115/donor_diversity")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.engine.resolve()))
    import final_model_engine as base
    import research_model_engine as engine
    from load_research_inputs import load_inputs

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    protocol_file = args.output / "PROTOCOL.json"
    planned = json.loads(json.dumps(design(args)))
    if args.action == "freeze":
        if protocol_file.exists():
            assert json.loads(protocol_file.read_text()) == planned
        else:
            base._atomic_json(protocol_file, planned)
        print(file_sha(protocol_file), flush=True)
        return
    assert json.loads(protocol_file.read_text()) == planned, "Frozen protocol or code hash mismatch"
    args_copy = args.output / ("inputs_" + (args.chain or args.action))
    raw, protocol, _ = load_inputs(args.data, args_copy)
    data = base._bundle(raw, {**base.DEFAULTS, "budgets": list(TARGET_BUDGETS)})
    if args.action == "check":
        reports = {}
        for chain in CHAINS:
            previous = set()
            for count in COUNTS:
                roster, labels, train = panel(data["donor_full"], chain, count)
                assert previous.issubset(roster)
                previous = set(roster)
                sampler = base.EpisodeSampler(train, [90], 14, 9, CHAINS[chain])
                episode_spans = {}
                for _ in range(25):
                    _, _, audit = sampler.sample()
                    assert audit["support_examples"] == 270 and audit["query_examples"] == 42
                    assert audit["purge_intervening_days"] >= 9
                    city = audit["city"]
                    episode_spans[city] = {
                        "support_days": (pd.Timestamp(audit["support_last"]) - pd.Timestamp(audit["support_first"])).days + 1,
                        "query_days": (pd.Timestamp(audit["query_last"]) - pd.Timestamp(audit["query_first"])).days + 1}
                reports[f"{chain}_{count}"] = {"cities": roster, "label_dates": len(labels),
                    "rows": len(train), "example_episode_calendar_spans": episode_spans,
                    "labels_by_year": labels.groupby(labels.date.dt.year).size().to_dict()}
        base._atomic_json(args.output / "PREFLIGHT.json", {"passed": True, "panels": reports,
            "target_cities": protocol["target_cities"], "gradient": engine.gradient_smoke(torch.device(args.device))})
    elif args.action == "run":
        if args.chain is None:
            parser.error("run requires --chain")
        run(args, base, engine, data)
    else:
        summarize(args, base, data)


if __name__ == "__main__":
    main()
