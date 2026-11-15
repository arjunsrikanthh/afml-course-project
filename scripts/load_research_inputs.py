"""Load hash-checked cached views without repeating raw XLSX I/O on a GPU."""
from pathlib import Path
import hashlib,json
import pandas as pd

def load_inputs(path,output):
    root,out=Path(path),Path(output)
    manifest=json.loads((root/'INPUT_MANIFEST.json').read_text())
    for name,entry in manifest.items():
        p=root/name
        assert p.stat().st_size==entry['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
    w=pd.read_csv(root/'forecast_examples.csv.gz',dtype={'station_id':str},
                  parse_dates=['date','issue_date','feature_date_min','feature_date_max'])
    cohort=pd.read_csv(root/'cohort.csv',dtype={'site_id':str})
    protocol=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'data_summary.json').read_text())
    targets=protocol['targets']
    donors=cohort[cohort.role.isin(['donor','donor_development'])].city.tolist()
    assert len(donors)==23 and len(targets)==6
    assert not set(donors)&set(targets)
    d=w[w.city.isin(donors)]
    bundle={'donor_train':d[(d.date<='2022-12-31')&(d.issue_date>='2020-01-01')].copy(),
      'donor_val':d[d.date.between('2023-01-04','2023-12-31')&(d.issue_date>='2023-01-01')].copy(),
      'donor_full':d[(d.date<='2023-12-31')&(d.issue_date>='2020-01-01')].copy(),
      'dev_cities':protocol['dev_cities'],'targets':{}}
    for city in targets:
        f=w[w.city==city]
        bundle['targets'][city]={'adapt':f[f.date<='2023-12-31'].copy(),
            'validation':f[f.date.between('2024-01-04','2024-12-31')&(f.issue_date>='2024-01-01')].copy(),
            'test':f[f.date.between('2025-01-01','2025-12-31')&(f.issue_date>='2024-01-01')].copy()}
    out.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in manifest:shutil.copy2(root/name,out/name)
    shutil.copy2(root/'INPUT_MANIFEST.json',out/'INPUT_MANIFEST.json')
    protocol['input_cache_manifest_sha256']=hashlib.sha256((root/'INPUT_MANIFEST.json').read_bytes()).hexdigest()
    protocol['target_cities']=targets
    protocol['expected_periods']=['secondary_2024','test_2025']
    protocol['latest_fit_cutoff']='2023-12-31'
    return bundle,protocol,summary
