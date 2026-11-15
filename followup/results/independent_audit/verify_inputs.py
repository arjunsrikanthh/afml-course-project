#!/usr/bin/env python3
"""Independent, read-only data/provenance guard for explanatory follow-up.

No training; no importing the production data selector or evaluator. Outputs are
restricted to this audit directory. Run from any working directory.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'results/final_study_20261114/mac_research/afml_research'
DATA = OLD / 'data'
checks = []
def check(name, condition, evidence):
    checks.append({'name': name, 'passed': bool(condition), 'evidence': evidence})
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

protocol = json.loads((OLD / 'frozen_protocol.json').read_text())
manifest = json.loads((DATA / 'INPUT_MANIFEST.json').read_text())
bad = [n for n, v in manifest.items()
       if (DATA/n).stat().st_size != v['bytes'] or sha(DATA/n) != v['sha256']]
check('original_input_manifest', not bad, {'files': len(manifest), 'mismatches': bad})
w = pd.read_csv(DATA / 'forecast_examples.csv.gz', dtype={'station_id':str},
                parse_dates=['date','issue_date','feature_date_min','feature_date_max'])
obs = pd.read_csv(DATA / 'observations.csv.gz', dtype={'station_id':str}, parse_dates=['date'])
cohort = pd.read_csv(DATA / 'cohort.csv', dtype={'site_id':str})
targets = sorted(protocol['targets'])
donors = cohort[cohort.role.isin(['donor','donor_development'])]
check('target_donor_exclusion', len(donors)==23 and not set(donors.city)&set(targets)
      and (donors.nearest_target_station_km >= 25).all(),
      {'donors':len(donors), 'targets':targets,
       'nearest_donor_to_target_km':float(donors.nearest_target_station_km.min())})
t = w[w.city.isin(targets)].copy()
check('independent_window_chronology',
      ((t.date-t.issue_date).dt.days==t.horizon).all()
      and (t.feature_date_min==t.issue_date-pd.Timedelta(days=6)).all()
      and (t.feature_date_max==t.issue_date).all()
      and (t['last']==t.x6).all()
      and not t.duplicated(['city','date','horizon']).any(), {'target_windows':len(t)})
lookup = obs.set_index(['station_id','date']).aqi
lag_failures = {}
for column, dates in [('y',t.date), *[(f'x{i}',t.issue_date-pd.Timedelta(days=6-i)) for i in range(7)]]:
    source = lookup.reindex(pd.MultiIndex.from_arrays([t.station_id, dates])).to_numpy(float)
    lag_failures[column] = int((~np.isclose(source,t[column].to_numpy(float),rtol=0,atol=1e-8)).sum())
check('independent_observed_truth_and_all_lags', not any(lag_failures.values()),lag_failures)

support_checks=[]
methods=['deep_gru_finetune','matched_gru_finetune','maml_gru_sgd','fomaml_gru_sgd','reptile_gru_sgd']
all_records=[json.loads(p.read_text()) for p in (OLD/'models/fit_records').glob('target_*.json')]
records={(r['city'],int(r['budget']),int(r['seed']),r['method']):r for r in all_records}
for city in targets:
    fit=t[(t.city==city)&(t.date<='2023-12-31')]
    for budget in [30,90,180]:
        dates=sorted(fit.date.unique())[-budget:]
        support=fit[fit.date.isin(dates)]
        values=[pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]
        for seed in [17,43,71]:
            for method in methods:
                r=records[(city,budget,seed,method)]
                ok=len(support)==3*budget and r['support_label_dates']==values and r['train_examples']==3*budget
                support_checks.append(ok)
check('independent_support_vs_original_records',all(support_checks),
      {'records':len(support_checks),'failures':len(support_checks)-sum(support_checks),
       'last_fitting_label':str(t[t.date<='2023-12-31'].date.max().date())})

snapshots={str(p.relative_to(ROOT)):sha(p) for p in sorted((OLD/'models/source_states').glob('*.pt'))}
for p in [OLD/'frozen_protocol.json',OLD/'models/frozen_adaptation_settings.json',
          ROOT/'delivery/submission_20261115/repository/scripts/research_model_engine.py',
          ROOT/'delivery/submission_20261115/repository/scripts/final_model_engine.py']:
    snapshots[str(p.relative_to(ROOT))]=sha(p)
snapshot_path=HERE/'original_hash_snapshot.json'
if snapshot_path.exists():
    before=json.loads(snapshot_path.read_text())
    check('original_artifacts_unchanged_since_independent_snapshot',before==snapshots,
          {'paths':len(snapshots),'changed':[k for k in set(before)|set(snapshots) if before.get(k)!=snapshots.get(k)]})
else:
    snapshot_path.write_text(json.dumps(snapshots,indent=2)+'\n')
    check('original_artifact_snapshot_created',True,{'paths':len(snapshots)})
report={'scope':'Inputs and original artifacts only; no new run completion claim',
        'passed':all(c['passed'] for c in checks),'checks':checks}
(HERE/'input_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
