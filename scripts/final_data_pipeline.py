"""Immutable CPCB extraction, descriptive EDA and frozen study construction."""
from pathlib import Path
import calendar, hashlib, importlib.util, json, platform, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = ['Amaravati', 'Guwahati', 'Ahmedabad', 'Pithampur', 'Ludhiana', 'Udaipur']
DEV_CITIES = ['Chandigarh', 'Dewas', 'Jalandhar', 'Ajmer']

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str))

def prepare(root, outdir):
    root, out = Path(root), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    register = pd.read_csv(root/'file_register.csv', dtype={'station_id':str})
    meta = pd.read_csv(root/'station_metadata.csv', dtype={'site_id':str})
    assert len(register)==3820 and len(meta)==626
    assert not meta.site_id.duplicated().any()
    spec = importlib.util.spec_from_file_location('release_parser', root/'scripts/build_paper_source_coverage.py')
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    rows, integrity = [], []
    for r in register.itertuples(index=False):
        path = root/'aqi_files'/r.filename
        assert path.stat().st_size==int(r.bytes), path
        digest = sha256(path)
        assert digest == r.sha256, path
        readings, layout, nrows, ncols = parser.parse_workbook(path, int(r.year))
        rows.extend((str(r.station_id), d, v) for d,v in readings.items())
        integrity.append({'filename':r.filename,'sha256':digest,'observations':len(readings),
                          'layout':f'{layout}_{nrows}x{ncols}',
                          'above_500':sum(v>500 for v in readings.values()),'below_zero':sum(v<0 for v in readings.values())})
    raw = pd.DataFrame(rows,columns=['station_id','date','aqi'])
    raw['date']=pd.to_datetime(raw.date)
    assert len(raw)==769129 and not raw.duplicated(['station_id','date']).any()
    assert np.isfinite(raw.aqi).all()
    raw=raw.sort_values(['station_id','date']).reset_index(drop=True)
    raw=raw.merge(meta.rename(columns={'site_id':'station_id'}),on='station_id',how='left',validate='many_to_one')
    pd.DataFrame(integrity).to_csv(out/'raw_integrity.csv',index=False)
    raw[['station_id','date','aqi','station_name','city','city_code','state']].to_csv(out/'observations.csv.gz',index=False)
    raw.loc[(raw.aqi>500)|(raw.aqi<0),['station_id','date','aqi','city']].to_csv(out/'range_flags.csv',index=False)
    raw.loc[raw.city.isna()].groupby('station_id').size().rename('observations').to_csv(out/'unmatched_ids.csv')
    matched=raw[raw.city.notna()].copy()
    matched['year']=matched.date.dt.year
    idx=pd.MultiIndex.from_product([meta.site_id.astype(str),range(2017,2026)],names=['station_id','year'])
    cov=matched.groupby(['station_id','year']).size().reindex(idx,fill_value=0).rename('observed').reset_index()
    cov['calendar_days']=cov.year.map(lambda y:366 if calendar.isleap(int(y)) else 365)
    cov['fraction']=cov.observed/cov.calendar_days
    cov=cov.merge(meta[['site_id','city','city_code','state','station_name']].rename(columns={'site_id':'station_id'}),on='station_id',validate='many_to_one')
    cov.to_csv(out/'station_year_coverage.csv',index=False)
    grid=cov.pivot(index='station_id',columns='year',values='fraction').reindex(columns=range(2020,2026))
    eligible_ids=grid.index[(grid>=.8).all(axis=1)]
    eligible=meta[meta.site_id.astype(str).isin(eligible_ids)].copy()
    rank=cov[cov.year.between(2020,2023)].groupby('station_id')['observed'].sum()
    eligible['training_observed_days']=eligible.site_id.map(rank)
    eligible['numeric_id']=eligible.site_id.astype(int)
    chosen=eligible.sort_values(['training_observed_days','numeric_id'],ascending=[False,True]).drop_duplicates('city_code')
    chosen=chosen.sort_values(['state','city']).drop(columns='numeric_id').reset_index(drop=True)
    assert len(chosen)==30 and set(TARGETS).issubset(chosen.city)
    chosen['role']=np.where(chosen.city.isin(TARGETS),'target',np.where(chosen.city.isin(DEV_CITIES),'donor_development','donor'))
    # Official city codes can subdivide one urban area (e.g. Ahmedabad/Vatva).
    # A declared25km station buffer removes adjacent aliases from source tasks.
    coords=np.radians(chosen[['latitude','longitude']].astype(float).to_numpy())
    target_geo=meta.loc[meta.city.isin(TARGETS),['latitude','longitude']].apply(pd.to_numeric,errors='coerce').dropna()
    tcoords=np.radians(target_geo.to_numpy())
    dist=[]
    for p in coords:
        a=np.sin((tcoords[:,0]-p[0])/2)**2+np.cos(p[0])*np.cos(tcoords[:,0])*np.sin((tcoords[:,1]-p[1])/2)**2
        dist.append(float((2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))).min()))
    chosen['nearest_target_station_km']=dist
    chosen.loc[(chosen.role!='target')&(chosen.nearest_target_station_km<25),'role']='excluded_source_buffer'
    donor_city_names=chosen.loc[chosen.role.isin(['donor','donor_development']),'city'].tolist()
    assert set(DEV_CITIES).issubset(donor_city_names)
    chosen.to_csv(out/'cohort.csv',index=False)
    eligible.drop(columns='numeric_id').to_csv(out/'eligible_stations.csv',index=False)
    selected=raw[raw.station_id.isin(chosen.site_id)].copy()
    win=[]
    for sid,group in selected.groupby('station_id',sort=True):
        group=group.set_index('date').sort_index()
        city=group.city.iloc[0]
        vals=group.aqi.to_dict()
        common=[]
        for d,y in vals.items():
            if not pd.Timestamp('2020-01-04')<=d<=pd.Timestamp('2025-12-31'):continue
            candidates=[]
            for h in (1,2,3):
                issue=d-pd.Timedelta(days=h)
                dates=pd.date_range(issue-pd.Timedelta(days=6),issue)
                if not all(t in vals for t in dates):break
                v=[vals[t] for t in dates]
                candidates.append({'station_id':sid,'city':city,'date':d,'issue_date':issue,'horizon':h,
                    'feature_date_min':dates[0],'feature_date_max':dates[-1],'y':float(y),'last':float(v[-1]),
                    **{f'x{i}':float(a) for i,a in enumerate(v)},'issue_doy':issue.dayofyear,'issue_dow':issue.dayofweek})
            if len(candidates)==3:common.extend(candidates)
        win.extend(common)
    windows=pd.DataFrame(win).sort_values(['city','date','horizon']).reset_index(drop=True)
    assert not windows.duplicated(['city','date','horizon']).any()
    assert (windows.feature_date_max==windows.issue_date).all()
    assert ((windows.date-windows.issue_date).dt.days==windows.horizon).all()
    assert (windows.feature_date_min==windows.issue_date-pd.Timedelta(days=6)).all()
    windows.to_csv(out/'forecast_examples.csv.gz',index=False)
    donors=windows[windows.city.isin(donor_city_names)]
    donor_train=donors[(donors.date<='2022-12-31')&(donors.issue_date>='2020-01-01')].copy()
    donor_val=donors[(donors.date.between('2023-01-04','2023-12-31'))&(donors.issue_date>='2023-01-01')].copy()
    donor_full=donors[(donors.date<='2023-12-31')&(donors.issue_date>='2020-01-01')].copy()
    targets={}
    counts=[]
    for city in TARGETS:
        d=windows[windows.city==city]
        adapt=d[d.date<='2023-12-31'].copy()
        secondary=d[d.date.between('2024-01-04','2024-12-31')&(d.issue_date>='2024-01-01')].copy()
        test=d[d.date.between('2025-01-01','2025-12-31')&(d.issue_date>='2024-01-01')].copy()
        assert adapt.date.nunique()>=180 and len(secondary)>0 and len(test)>0
        targets[city]={'adapt':adapt,'validation':secondary,'test':test}
        for period,frame in [('adapt',adapt),('secondary_2024',secondary),('test_2025',test)]:
            counts.append({'city':city,'station_id':str(frame.station_id.iloc[0]),'period':period,
                           'unique_label_dates':frame.date.nunique(),'examples':len(frame),
                           'first_label':frame.date.min(),'last_label':frame.date.max(),
                           'eligible_fraction_of_calendar':frame.date.nunique()/(366 if period=='secondary_2024' else 365) if period!='adapt' else None})
    pd.DataFrame(counts).to_csv(out/'eligibility.csv',index=False)
    # Descriptive EDA: value inspection confined to development years before model evaluation.
    dev=raw[raw.date.dt.year.between(2020,2023)].copy()
    dev['year']=dev.date.dt.year
    dev['month']=dev.date.dt.month
    dev.aqi.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_csv(out/'development_distribution.csv')
    dev.groupby('year').agg(observations=('aqi','size'),stations=('station_id','nunique'),mean=('aqi','mean'),median=('aqi','median')).to_csv(out/'development_by_year.csv')
    dev[dev.city.notna()].groupby(['city','month']).aqi.agg(['mean','median','count']).to_csv(out/'development_seasonality.csv')
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    axs[0].hist(dev.aqi,bins=60,color='#2b7b9f');axs[0].set(title='Observed AQI, 2020–2023',xlabel='AQI',ylabel='Station-days')
    coverage_image=cov.pivot(index='station_id',columns='year',values='fraction').sort_values(list(range(2017,2026)),ascending=False)
    im=axs[1].imshow(coverage_image.values,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    axs[1].set(title='Calendar observation coverage: current roster',xlabel='Year',ylabel='Station (sorted)')
    axs[1].set_xticks(range(9),range(2017,2026),rotation=45);fig.colorbar(im,ax=axs[1],label='Fraction observed')
    fig.tight_layout();fig.savefig(out/'eda_source.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for city in TARGETS:
        s=dev[dev.city==city].groupby('month').aqi.mean()
        ax.plot(s.index,s.values,marker='o',label=city)
    ax.set(title='Development-period target-city seasonality (observed stations)',xlabel='Month',ylabel='Mean AQI');ax.legend(ncol=3)
    fig.tight_layout();fig.savefig(out/'eda_seasonality.png',dpi=160);plt.close(fig)
    autoc=[]
    for sid,g in selected.groupby('station_id'):
        s=g[g.date.dt.year.between(2020,2023)].set_index('date').aqi.asfreq('D')
        autoc.append({'station_id':sid,'city':g.city.iloc[0],**{f'lag_{l}':s.corr(s.shift(l)) for l in [1,2,3,7,14]}})
    pd.DataFrame(autoc).to_csv(out/'development_autocorrelation.csv',index=False)
    summary={'workbooks':len(register),'observations':len(raw),'matched_observations':int(raw.city.notna().sum()),
             'unmatched_observations':int(raw.city.isna().sum()),'empty_workbooks':sum(r['observations']==0 for r in integrity),
             'above500':int((raw.aqi>500).sum()),'duplicates':0,'parse_errors':0,'eligible_stations_at_least_80pct':len(eligible),
             'representative_cities':len(chosen),'donor_cities':donors.city.nunique(),'target_cities':TARGETS,'development_tuning_cities':DEV_CITIES,
             'integrity_seconds':time.time()-started,'python':platform.python_version(),
             'register_sha256':sha256(root/'file_register.csv'),'metadata_sha256':sha256(root/'station_metadata.csv')}
    write_json(out/'data_summary.json',summary)
    protocol={'title':'Cross-City Transfer Learning for AQI Forecasting in Data-Scarce Indian Regions',
      'design':'retrospective, frozen before this run; old pilot results and broad coverage already inspected',
      'source_release_sha256':'014fa6ab88854a9649e4d903483d516be5b4b5ecf482ecd497e291cdf1a89811',
      'source_period':'2017–2025; modeling 2020–2025', 'targets':TARGETS,'dev_cities':DEV_CITIES,
      'station_selection':'at least 80% observed each2020–2025; one per city_code ranked by2020–2023 observed days, numeric ID tie-break',
      'target_selection':'six untested pilot cities in different states, chosen before forecast inspection; convenience availability cohort',
      'source_geography_guard':'exclude any non-target representative station within25km of ANY known station in a target city, including Vatva beside Ahmedabad; current coordinates, not a historic boundary proof',
      'budgets':[30,90,180],'budget_definition':'B unique most-recent eligible target label dates <=2023-12-31, shared across3horizons (3B training pairs); preceding input context additionally available',
      'input':'7 consecutive observed daily AQI values, issue-calendar features, horizon; no weather/peers/future inputs',
      'forecast':'direct horizon-conditioned residual output + last AQI; continuous uncapped',
      'horizons':[1,2,3],'donor_train':'2020–2022','donor_tune':'2023, development cities excluded preliminary fitting',
      'donor_refit':'2020–2023 after donor-only setting selection','target_fitting_cutoff':'2023-12-31',
      'secondary_evaluation':'2024Jan4–Dec31, origins>=2024Jan1; never used for tuning',
      'main_evaluation':'2025 all common eligible label dates; frozen2023weights with rolling observed inputs',
      'exclusions':'unmatched identities and insufficient coverage; gaps yield ineligible windows; no imputation of inputs or truth',
      'seeds':[17,43,71],'metrics':['MAE','RMSE','bias','persistence-relative skill'],
      'uncertainty':'seed-average paired losses; within-city14calendar-day block bootstrap, macro6city mean; conditional on these6cities',
      'limitations':['station not city average','simulated fitting-label scarcity not naturally sparse monitoring','last-B selection confounds recency and season',
        'test availability used for eligibility, not performance','target cities convenience sample','frozen2023model ages through2025',
        'daily nominal1–3days approximate24–72hours; publication latency not verified','CPCB current roster not historic census']}
    write_json(out/'protocol.json',protocol)
    return {'donor_train':donor_train,'donor_val':donor_val,'donor_full':donor_full,'targets':targets,'dev_cities':DEV_CITIES},protocol,summary
