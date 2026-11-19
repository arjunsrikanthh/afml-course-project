#!/usr/bin/env python3
"""Stage and execute the frozen checkpoint-based AQI follow-up experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

from reproduce import ROOT, sha256


FOLLOWUP = ROOT / "followup"
STAGE = ROOT / "runs/followup"
FROZEN_REL = Path("delivery/submission_20261115/repository")
MODELS_REL = Path("results/final_study_20261114/mac_research/afml_research/models")
RESULTS_REL = Path("results/explanatory_investigation_20261115")
SCRIPTS_REL = Path("scripts/investigation_20261115")
AUDIT_REL = RESULTS_REL / "independent_audit"


def read_json(path):
    return json.loads(path.read_text())


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def archive_path(local, release, asset):
    if local is not None:
        path = local.resolve()
    else:
        path = ROOT / "artifacts" / asset["name"]
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            url = f"https://github.com/{release['repository']}/releases/download/{release['release']}/{asset['name']}"
            print(f"Downloading {url}", flush=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                temporary = Path(temp.name)
            try:
                with urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                verify(temporary, asset["sha256"])
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    if path.stat().st_size != asset["bytes"]:
        raise ValueError(f"Archive size mismatch: {path}")
    verify(path, asset["sha256"])
    return path


def stage_member(handle, member, destination, expected_sha256):
    if destination.is_file() and sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with handle.open(member) as source, temporary.open("wb") as stream:
            shutil.copyfileobj(source, stream)
        verify(temporary, expected_sha256)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def protocol_files():
    return {
        "adaptation": (FOLLOWUP / "protocols/adaptation.json", STAGE / RESULTS_REL / "adaptation/protocol.json"),
        "meta_initialization": (
            FOLLOWUP / "protocols/meta_initialization.json",
            STAGE / RESULTS_REL / "meta_initialization/PROTOCOL.json",
        ),
        "donor_diversity": (
            FOLLOWUP / "protocols/donor_diversity.json",
            STAGE / RESULTS_REL / "donor_diversity/PROTOCOL.json",
        ),
    }


def copy_locked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination, sha256(source))
    else:
        shutil.copyfile(source, destination)


def stage_original_audit_inputs(handle):
    prefix = "AFML_Complete_Project/expanded/"
    snapshot = read_json(FOLLOWUP / "results/independent_audit/original_hash_snapshot.json")
    for relative, expected in snapshot.items():
        path = Path(relative)
        if path.is_relative_to(MODELS_REL.parent):
            name = path.relative_to(MODELS_REL.parent).as_posix()
            stage_member(handle, prefix + name, STAGE / path, expected)
        else:
            verify(STAGE / path, expected)

    manifest_member = prefix + "data/INPUT_MANIFEST.json"
    with handle.open(manifest_member) as manifest_file:
        manifest_digest = hashlib.file_digest(manifest_file, "sha256").hexdigest()
    original_data = STAGE / MODELS_REL.parent / "data"
    stage_member(handle, manifest_member, original_data / "INPUT_MANIFEST.json", manifest_digest)
    manifest = read_json(original_data / "INPUT_MANIFEST.json")
    for name, entry in manifest.items():
        stage_member(handle, prefix + "data/" + name, original_data / name, entry["sha256"])

    # Only fitting journals referenced by the independent input audit are needed.
    methods = ("deep_gru_finetune", "matched_gru_finetune", "maml_gru_sgd", "fomaml_gru_sgd", "reptile_gru_sgd")
    suffixes = tuple(f"_{method}.json" for method in methods)
    fit_prefix = prefix + "models/fit_records/target_"
    records = [name for name in handle.namelist() if name.startswith(fit_prefix) and name.endswith(suffixes)]
    if len(records) != 270:
        raise ValueError(f"Expected 270 original target fitting journals, found {len(records)}")
    for member in records:
        destination = STAGE / MODELS_REL / "fit_records" / Path(member).name
        with handle.open(member) as record_file:
            expected = hashlib.file_digest(record_file, "sha256").hexdigest()
        stage_member(handle, member, destination, expected)


def prepare(args):
    reference_release = read_json(ROOT / "manifests/release-assets.json")
    study_name = "AFML_Complete_Executed_Notebooks.zip"
    study_asset = {"name": study_name, **reference_release["assets"][study_name]}
    reference_archive = archive_path(args.study_archive, reference_release, study_asset)
    replay = read_json(FOLLOWUP / "replay-inputs.json")
    replay_archive = archive_path(args.replay_archive, replay, replay["asset"])

    frozen = STAGE / FROZEN_REL
    frozen.mkdir(parents=True, exist_ok=True)
    for name, target in {"scripts": ROOT / "scripts", "data": ROOT / "data"}.items():
        link = frozen / name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if not link.is_symlink() or link.resolve() != target:
            raise ValueError(f"Unexpected frozen input path: {link}")
    for source in sorted((FOLLOWUP / "scripts").glob("*.py")):
        copy_locked(source, STAGE / SCRIPTS_REL / source.name)
    for name in ("verify_inputs.py", "verify_outputs.py", "original_hash_snapshot.json"):
        copy_locked(FOLLOWUP / "results/independent_audit" / name, STAGE / AUDIT_REL / name)

    protocols = {name: read_json(source) for name, (source, _) in protocol_files().items()}
    adaptation = protocols["adaptation"]
    meta = protocols["meta_initialization"]
    for name, expected in adaptation["frozen_engine_sha256"].items():
        verify(frozen / "scripts" / name, expected)
    verify(frozen / "data/frozen/INPUT_MANIFEST.json", adaptation["input_manifest_sha256"])
    verify(STAGE / SCRIPTS_REL / "adaptation_controls.py", adaptation["investigation_script_sha256"])

    models = STAGE / MODELS_REL
    model_members = dict(adaptation["source_states_sha256"])
    for name in ("effective_config.json", "frozen_adaptation_settings.json"):
        relative = (MODELS_REL / name).as_posix()
        model_members[relative] = meta["source_sha256"][relative]
    with ZipFile(reference_archive) as handle:
        stage_original_audit_inputs(handle)
        for name, expected in replay["additional_original_model_files_sha256"].items():
            stage_member(handle, f"AFML_Complete_Project/expanded/models/{name}", models / name, expected)
        for relative, expected in model_members.items():
            name = Path(relative).relative_to(MODELS_REL).as_posix()
            stage_member(handle, f"AFML_Complete_Project/expanded/models/{name}", models / name, expected)
    with ZipFile(replay_archive) as handle:
        if sorted(handle.namelist()) != sorted(replay["files_sha256"]):
            raise ValueError("Unexpected replay ZIP members")
        for name, expected in replay["files_sha256"].items():
            stage_member(handle, name, models / name, expected)
    for relative, expected in meta["source_sha256"].items():
        verify(STAGE / relative, expected)
    donor = protocols["donor_diversity"]["source_sha256"]
    verify(STAGE / SCRIPTS_REL / "donor_diversity_controls.py", donor["experiment"])
    for source, destination in protocol_files().values():
        copy_locked(source, destination)
    print(f"Verified and staged frozen follow-up inputs at {STAGE}", flush=True)


def run(script, *args):
    path = STAGE / SCRIPTS_REL / script
    verify(path, sha256(FOLLOWUP / "scripts" / script))
    subprocess.run([sys.executable, str(path), *args], cwd=STAGE, check=True)


def audit(*args):
    for filename, *options in args:
        path = STAGE / AUDIT_REL / filename
        verify(path, sha256(FOLLOWUP / "results/independent_audit" / filename))
        subprocess.run([sys.executable, str(path), *options], cwd=STAGE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "prepare", "smoke", "adaptation", "meta", "summarize",
        "donor-check", "donor-run", "donor-summarize", "audit-inputs", "audit",
    ])
    parser.add_argument("--study-archive", type=Path, help="Existing original release ZIP")
    parser.add_argument("--replay-archive", type=Path, help="Existing follow-up replay-inputs ZIP")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu",
                        help="Adaptation or donor-panel device. Meta replay uses CPU.")
    parser.add_argument("--chain", choices=["a", "b", "c"], help="Chain for donor-run")
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads for donor commands")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args)
        return
    if not (STAGE / MODELS_REL / "effective_config.json").is_file():
        parser.error("Run python reproduce_followup.py prepare first")
    if args.command == "smoke":
        run("adaptation_controls.py", "smoke", "--device", args.device)
        run("adaptation_controls.py", "parity", "--device", args.device)
        run("meta_initialization_controls.py", "--freeze")
    elif args.command == "adaptation":
        run("adaptation_controls.py", "run", "--device", args.device)
    elif args.command == "meta":
        run("meta_initialization_controls.py", "--run")
    elif args.command == "summarize":
        run("summarize_explanation.py")
    elif args.command == "audit-inputs":
        audit(("verify_inputs.py",))
    elif args.command == "audit":
        audit(("verify_inputs.py",), ("verify_outputs.py", "adaptation"), ("verify_outputs.py", "meta"))
    else:
        action = args.command.removeprefix("donor-")
        options = ["--device", args.device, "--threads", str(args.threads)]
        if action == "run":
            if args.chain is None:
                parser.error("donor-run requires --chain")
            options.extend(["--chain", args.chain])
        run("donor_diversity_controls.py", action, *options)
        if action == "summarize":
            run("verify_donor_diversity.py")
            run("summarize_donor_composition.py", str(STAGE / RESULTS_REL / "donor_diversity"))


if __name__ == "__main__":
    main()
