#!/usr/bin/env python3
"""Portable entry point for the frozen daily-AQI study. Run --help for commands."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
PROJECT = ARTIFACTS / 'AFML_Complete_Project'
SOURCE = ARTIFACTS / 'CPCB_AQI_Source_Release_v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        if handle.testzip() is not None:
            raise ValueError(f'Corrupt ZIP: {archive}')
        handle.extractall(destination)


def fetch(args):
    config = json.loads((ROOT / 'manifests/release-assets.json').read_text())
    name = 'AFML_Complete_Executed_Notebooks.zip'
    expected = config['assets'][name]
    ARTIFACTS.mkdir(exist_ok=True)
    archive = ARTIFACTS / name
    if not archive.exists() or sha256(archive) != expected['sha256']:
        url = f"https://github.com/{config['repository']}/releases/download/{config['release']}/{name}"
        temporary = archive.with_suffix('.download')
        print(f'Downloading public frozen study: {url}', flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if temporary.stat().st_size != expected['bytes'] or sha256(temporary) != expected['sha256']:
            temporary.unlink(missing_ok=True)
            raise ValueError('Release asset size/SHA256 differs from the frozen manifest')
        temporary.replace(archive)
    extract(archive, ARTIFACTS)
    source_archive = PROJECT / 'source_data/CPCB_AQI_Source_Release_v1.zip'
    if sha256(source_archive) != config['assets'][source_archive.name]['sha256']:
        raise ValueError('CPCB source snapshot SHA256 differs')
    extract(source_archive, ARTIFACTS)
    print(f'Verified and extracted: {PROJECT}\nSource: {SOURCE}')


def check_inputs(args):
    from scripts.load_research_inputs import load_inputs
    output = ROOT / 'runs/input_check'
    bundle, protocol, summary = load_inputs(ROOT / 'data/frozen', output)
    print(json.dumps({'verified_manifest_files': len(json.loads((ROOT / 'data/frozen/INPUT_MANIFEST.json').read_text())),
        'donor_cities': bundle['donor_full'].city.nunique(), 'target_cities': list(bundle['targets']),
        'source_workbooks': summary['workbooks'], 'observations': summary['observations']}, indent=2))


def prepare(args):
    from scripts.final_data_pipeline import prepare as prepare_data
    if not SOURCE.exists():
        raise FileNotFoundError('Run python reproduce.py fetch first (public source archive required)')
    out = ROOT / 'runs/prepared_data'
    prepare_data(SOURCE, out)
    # gzip headers and elapsed-time metadata vary. Compare the scientific tables,
    # not byte serialization that contains a new timestamp.
    import pandas as pd
    for name in ['observations.csv.gz', 'forecast_examples.csv.gz', 'cohort.csv', 'eligibility.csv', 'station_year_coverage.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/name), pd.read_csv(ROOT/'data/frozen'/name))
    print(f'Reparsed all 3,820 source workbooks; five scientific tables match the frozen inputs: {out}')


def smoke(args):
    from scripts.final_model_engine import smoke_test as baseline_smoke
    from scripts.research_model_engine import smoke_test as final_smoke
    output = ROOT / 'runs/smoke'
    print(json.dumps({'baseline': baseline_smoke(output/'baseline'),
                      'final': final_smoke(output/'final')}, indent=2))


def train(args):
    from scripts.load_research_inputs import load_inputs
    run = Path(args.output).resolve() if args.output else ROOT / 'runs' / args.study
    bundle, protocol, summary = load_inputs(ROOT/'data/frozen', run/'data')
    config_path = (ROOT/'configs/archive/reference.json' if args.study == 'baseline'
                   else ROOT/'configs/final.json')
    config = json.loads(config_path.read_text())
    config['device'] = args.device
    historical = json.loads((ROOT/'results'/args.study/'frozen_protocol.json').read_text())
    protocol.update({k: v for k, v in historical.items() if k not in protocol})
    protocol['model_config'] = config
    protocol['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    frozen_path = run/'frozen_protocol.json'
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        protocol['frozen_utc'] = previous['frozen_utc']
        if previous != json.loads(json.dumps(protocol, default=str)):
            raise ValueError('Resume protocol/config changed; select a new --output directory')
    else:
        frozen_path.write_text(json.dumps(protocol, indent=2, default=str)+'\n')
    if args.study == 'baseline':
        from scripts.final_model_engine import train_all
        from scripts.final_evaluation import evaluate
        predictions, fits = train_all(bundle, run/'models', config)
        protocol['fit_costs'] = fits
        evaluate(predictions, run/'evaluation', protocol)
    else:
        from scripts.research_model_engine import train_research
        from scripts.research_analysis import analyze_research
        predictions, fits = train_research(bundle, run/'models', config)
        reference = ROOT/'runs/baseline'
        analyze_research(predictions, fits, run/'evaluation', protocol,
                         compact_root=reference if (reference/'evaluation/macro_metrics.csv').exists() else None)
    print(f'Completed {args.study}: {len(predictions):,} predictions; {len(fits):,} fitting records; {run}')


def audit(args):
    from scripts.verify_final_study import verify_study
    from scripts.verify_research_study import verify_research_study
    if not SOURCE.exists():
        raise FileNotFoundError('Run python reproduce.py fetch first')
    targets = [('baseline',verify_study),('final',verify_research_study)] if args.study=='both' else [(args.study, verify_study if args.study=='baseline' else verify_research_study)]
    archive_names = {'baseline': 'reference', 'final': 'expanded'}
    for name, verifier in targets:
        run = Path(args.root).resolve() if args.root else PROJECT/archive_names[name]
        output = ROOT/'runs/audits'/name
        result = verifier(run, output, SOURCE)
        print(f"{name}: {result['verdict']} ({len(result['checks'])} checks), {output}")


def synthesis(args):
    import nbformat
    from nbclient import NotebookClient
    inputs = Path(args.inputs).resolve() if args.inputs else PROJECT
    output = Path(args.output).resolve() if args.output else ROOT/'runs/synthesis'
    os.environ['AFML_SYNTHESIS_INPUTS'] = str(inputs)
    os.environ['AFML_SYNTHESIS_OUTPUT'] = str(output)
    notebook = nbformat.read(ROOT/'notebooks/runnable/04_synthesis.ipynb', as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    output.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output/'04_synthesis_executed.ipynb')
    verification = json.loads((output/'synthesis_verification.json').read_text())
    print(json.dumps(verification, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command', required=True)
    for command, help_text, handler in [
        ('fetch','Download, SHA256-verify and extract the public frozen source/results archive',fetch),
        ('check-inputs','Verify every cached input hash and reconstruct legal donor/target splits',check_inputs),
        ('prepare','Reparse all original workbooks and compare scientific tables with frozen inputs',prepare),
        ('smoke','Artificial baseline/final training, resume, leakage and derivative checks',smoke)]:
        p=commands.add_parser(command,help=help_text);p.set_defaults(handler=handler)
    p=commands.add_parser('train',help='Fit the baseline or final benchmark; the final benchmark takes hours')
    p.add_argument('--study',choices=['baseline','final'],required=True)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--output');p.set_defaults(handler=train)
    p=commands.add_parser('audit',help='Independently audit saved forecasts without fitting models')
    p.add_argument('--study',choices=['baseline','final','both'],default='both')
    p.add_argument('--root',help='Explicit run root; select one --study with this option');p.set_defaults(handler=audit)
    p=commands.add_parser('synthesis',help='Execute the final research synthesis from saved audited forecasts')
    p.add_argument('--inputs');p.add_argument('--output');p.set_defaults(handler=synthesis)
    args=parser.parse_args()
    if args.command=='audit' and args.root and args.study=='both':parser.error('--root requires --study baseline or final')
    args.handler(args)

if __name__=='__main__':main()
