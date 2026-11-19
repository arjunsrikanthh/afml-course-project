#!/usr/bin/env python3
"""Reproduce the cross-architecture and MAML-objective extension."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import pandas as pd

import reproduce_followup as replay


EXTENSION = replay.ROOT / "followup" / "extension"
CROSS = replay.STAGE / "results/cross_architecture_followup_20261115"
MAML = replay.STAGE / "results/explanatory_extension_20261115/maml_objective"
MODELS = replay.STAGE / replay.MODELS_REL
ARCHITECTURES = ("lstm", "mlp", "tcn", "transformer")
METHODS = {f"deep_{name}_{kind}" for name in ARCHITECTURES for kind in ("pooled_zero", "finetune")}
ARCHIVE_PREFIX = "AFML_Complete_Project/expanded/models/"


def rebuild_archive_blocks(archive):
    """Split the pinned consolidated archive while retaining decimal strings."""
    key = ["city", "budget", "seed", "method"]
    found = set()
    with tempfile.TemporaryDirectory(dir=MODELS) as temporary:
        target = Path(temporary)
        with archive.open(ARCHIVE_PREFIX + "predictions.csv.gz") as handle:
            for frame in pd.read_csv(handle, compression="gzip", dtype=str, chunksize=150_000):
                for (city, budget, seed, method), rows in frame[frame.method.isin(METHODS)].groupby(key):
                    city_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", city).strip("_")
                    city_hash = hashlib.sha256(city.encode()).hexdigest()[:8]
                    name = f"{city_slug}_{city_hash}_b{budget}_s{seed}_{method}.csv.gz"
                    path = target / name
                    rows.to_csv(path, index=False, header=name not in found, mode="a", compression="gzip")
                    found.add(name)
        if len(found) != 432:
            raise ValueError(f"Expected 432 archived model blocks, found {len(found)}")
        block_dir = MODELS / "prediction_blocks"
        block_dir.mkdir(exist_ok=True)
        for source in target.iterdir():
            shutil.move(source, block_dir / source.name)


def prepare(args):
    replay.prepare(args)
    for path in sorted((EXTENSION / "scripts").glob("*.py")):
        replay.copy_locked(path, replay.STAGE / replay.SCRIPTS_REL / path.name)
    protocols = {
        "cross_architecture.json": CROSS / "protocol.json",
        "maml_objective.json": MAML / "PROTOCOL.json",
        "maml_broad_query.json": MAML / "broad_query/PROTOCOL.json",
    }
    for name, output in protocols.items():
        replay.copy_locked(EXTENSION / "protocols" / name, output)
    cross = replay.read_json(CROSS / "protocol.json")
    paired = replay.read_json(MAML / "PROTOCOL.json")
    hashes = dict(cross["source_states_sha256"])
    hashes.update({name: digest for name, digest in paired["sha256"].items()
                   if Path(name).is_relative_to(replay.MODELS_REL)})
    release = replay.read_json(replay.ROOT / "manifests/release-assets.json")
    name = "AFML_Complete_Executed_Notebooks.zip"
    path = replay.archive_path(args.study_archive, release, {"name": name, **release["assets"][name]})
    with ZipFile(path) as archive:
        for name, digest in hashes.items():
            relative = Path(name).relative_to(replay.MODELS_REL).as_posix()
            destination = replay.STAGE / name
            if relative.startswith("checkpoints/"):
                replay.verify(destination, digest)
            else:
                replay.stage_member(archive, ARCHIVE_PREFIX + relative, destination, digest)
        rebuild_archive_blocks(archive)
    print(f"Verified and staged extension inputs at {replay.STAGE}", flush=True)


def run(script, *args):
    source = EXTENSION / "scripts" / script
    staged = replay.STAGE / replay.SCRIPTS_REL / script
    replay.verify(staged, replay.sha256(source))
    subprocess.run([sys.executable, str(staged), *args], cwd=replay.STAGE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "prepare", "cross-smoke", "cross-run", "cross-audit", "maml-freeze", "maml-run", "maml-broad", "maml-audit",
    ])
    parser.add_argument("--study-archive", type=Path, help="Optional local original release ZIP")
    parser.add_argument("--replay-archive", type=Path, help="Optional local checkpoint supplement ZIP")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Cross-architecture device")
    parser.add_argument("--seed", type=int, choices=[17, 43, 71], help="One seed for maml-run")
    args = parser.parse_args()
    command = args.command
    if command == "prepare":
        prepare(args)
    elif command in ("cross-smoke", "cross-run"):
        action = "parity" if command == "cross-smoke" else "run"
        run("cross_architecture_controls.py", action, "--device", args.device)
    elif command == "cross-audit":
        run("verify_cross_architecture.py")
        run("summarize_cross_architecture.py")
    elif command == "maml-freeze":
        run("maml_objective_controls.py", "freeze")
        run("maml_query_sampling_control.py", "freeze")
    elif command == "maml-run":
        if args.seed is None:
            parser.error("maml-run requires --seed")
        run("maml_objective_controls.py", "run", "--seed", str(args.seed))
    elif command == "maml-broad":
        run("maml_query_sampling_control.py", "run")
    else:
        run("verify_maml_objectives.py")


if __name__ == "__main__":
    main()
