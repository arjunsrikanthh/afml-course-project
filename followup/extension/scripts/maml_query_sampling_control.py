#!/usr/bin/env python3
"""Hold the zero-step objective fixed and change donor query-date sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import maml_objective_controls as paired


NAME = "zero_broad_query_objective"
BROAD_BATCH_SEED_OFFSET = 17137


def freeze(out):
    paired_hash = paired.protocol(out.parent)
    expected = {
        "stage": "post hoc on previously inspected target years; no target selection",
        "paired_protocol_sha256": paired_hash,
        "script_sha256": paired.digest(Path(__file__)),
        "arm": NAME,
        "estimand": (
            "Change only the zero-step objective's query sampling. Instead of "
            "sampling 32 unique examples from the chosen task's 14 query-label dates, "
            "choose 32 unique examples uniformly from all eligible source-refit rows "
            "of that task's city. Keep the exact archived task city stream, architecture, "
            "initialization, optimizer, 1350 outer updates, 4 query batches/update, "
            "32 rows/batch, loss, and clipping. Donor chronology respects the 2023 cutoff."
        ),
        "rng_seed_offset": BROAD_BATCH_SEED_OFFSET,
        "evaluation_and_stages": "identical to paired protocol",
        "meaning": (
            "Isolates procedural within-city query sampling, including the change "
            "from localized query blocks to all eligible dates. A difference "
            "cannot isolate chronology from temporal concentration or date weights."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "PROTOCOL.json"
    if path.exists():
        assert json.loads(path.read_text()) == expected
    else:
        paired.base._atomic_json(path, expected)
    return paired.digest(path)


def run(seed, data, config, out, protocol_hash):
    began = time.monotonic()
    train = data["donor_full"].reset_index(drop=True)
    assert train.city.nunique() == 23 and len(train) == 71829
    assert train.date.max() <= pd.Timestamp("2023-12-31") and set(train.city).isdisjoint(data["targets"])
    device = torch.device("cpu")
    arrays = paired.base._arrays(train, device)
    city_rows = {city: group.index.to_numpy() for city, group in train.groupby("city")}
    expected = list(pd.read_csv(paired.journal(seed)).itertuples(index=False))
    preset = paired.json_config(paired.ORIGINAL / "source_selections.json")["meta"]["maml"]
    hyper = preset["preset"]
    assert (hyper["inner_steps"], hyper["inner_lr"], hyper["outer_lr"], preset["steps"]) == (5, .03, .0003, 1350)
    network = paired.model(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=hyper["outer_lr"])
    sampler = paired.base.EpisodeSampler(train, config["budgets"], config["meta_query_days"],
                                        config["purge_days"], seed + 177)
    rng = np.random.default_rng(seed + paired.BASE_SEED)
    broad_rng = np.random.default_rng(seed + BROAD_BATCH_SEED_OFFSET)
    epi_counts = np.zeros(len(train), dtype=np.int64)
    broad_counts = np.zeros(len(train), dtype=np.int64)
    for step in range(1, preset["steps"] + 1):
        totals = [torch.zeros_like(parameter) for parameter in network.parameters()]
        network.train()
        for task in range(config["meta_tasks"]):
            si, qi, episode = sampler.sample()
            for _ in range(hyper["inner_steps"]):
                rng.choice(si, size=min(config["meta_batch_size"], len(si)), replace=True)
            episodic_query = rng.choice(qi, size=min(config["meta_batch_size"], len(qi)), replace=False)
            recorded = expected[(step-1)*config["meta_tasks"]+task]
            paired.validate_journal(episode, si, qi, recorded, step, task, arrays.dates)
            rows = city_rows[episode["city"]]
            broad_query = broad_rng.choice(rows, size=config["meta_batch_size"], replace=False)
            epi_counts[episodic_query] += 1
            broad_counts[broad_query] += 1
            gradients, _ = paired.engine._meta_gradient(network, arrays.x, arrays.residual, [],
                broad_query, hyper, "maml", float(config["grad_clip"]))
            for total, gradient in zip(totals, gradients):
                total.add_(gradient)
        optimizer.zero_grad(set_to_none=True)
        for parameter, total in zip(network.parameters(), totals):
            parameter.grad = total / config["meta_tasks"]
        torch.nn.utils.clip_grad_norm_(network.parameters(), float(config["grad_clip"]))
        optimizer.step()
        if step in paired.SOURCE_STAGES:
            paired.base._torch_save(out / f"trained_s{seed}_step{step}.pt", {NAME: paired.base._state(network)})
            paired.evaluate({NAME: network}, seed, step, data, config, out)
            print(f"seed={seed} stage={step} evaluated elapsed={time.monotonic()-began:.1f}s", flush=True)
    released = paired.base._torch_load(
        paired.ORIGINAL / "checkpoints" / f"refit_meta_maml_s{seed}.pt", device)
    assert rng.bit_generator.state == released["inner_rng"]
    assert sampler.rng.bit_generator.state == released["sampler_rng"]
    counts = train[["city", "station_id", "date", "issue_date", "horizon"]].assign(
        episodic_query_draws=epi_counts, broad_query_draws=broad_counts)
    assert epi_counts.sum() == broad_counts.sum() == 1350 * 4 * 32
    city_counts = counts.groupby("city")[["episodic_query_draws", "broad_query_draws"]].sum()
    assert (city_counts.episodic_query_draws == city_counts.broad_query_draws).all()
    paired.base._atomic_csv(out / f"query_draws_s{seed}.csv.gz", counts)
    paired.base._atomic_json(out / f"receipt_s{seed}.json", {
        "protocol_sha256": protocol_hash, "seed": seed, "archived_final_rngs_exact": True,
        "matched_city_draw_totals": True, "total_rows_per_arm": int(epi_counts.sum()),
        "episodic_distinct_query_rows": int(np.count_nonzero(epi_counts)),
        "broad_distinct_query_rows": int(np.count_nonzero(broad_counts)),
        "elapsed_seconds": time.monotonic()-began,
        "sha256": {path.name: paired.digest(path) for path in out.glob(f"*_s{seed}*") if path.is_file()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=paired.DEFAULT_OUT / "broad_query")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_hash = freeze(args.out)
    if args.action == "freeze":
        print(protocol_hash)
        return
    config = paired.json_config(paired.ORIGINAL / "effective_config.json")
    config["_deadline"] = float("inf")
    data = paired.load_inputs(paired.FROZEN / "data/frozen", args.out / "inputs")[0]
    for seed in paired.SEEDS:
        run(seed, data, config, args.out, protocol_hash)


if __name__ == "__main__":
    main()
