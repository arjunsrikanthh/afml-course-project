#!/usr/bin/env python3
"""Post-hoc checkpoint-conditioned initialization/adaptation diagnostic; CPU only."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'delivery/submission_20261115/repository'
sys.path.insert(0, str(FROZEN / 'scripts'))
# Import exclusively the released modules, not the dirty root scripts package.
import final_model_engine as base
import research_model_engine as engine
from load_research_inputs import load_inputs
OUT = ROOT / 'results/explanatory_investigation_20261115/meta_initialization'
OLD = ROOT / 'results/final_study_20261114/mac_research/afml_research/models'
DEVICE = torch.device('cpu')
INITS = ['pooled', 'maml', 'fomaml', 'reptile']
STEPS = [1, 3, 10, 30]
LRS = [.01, .03]
SEEDS = [17, 43, 71]
BUDGETS = [30, 90, 180]
SPEC = {'architecture': 'manual_gru', 'hidden': 64, 'layers': 2}
DONOR_PANEL = None

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, obj):
    base._atomic_json(path, obj)

def checkpoint(init, seed):
    stem = 'source_manual_gru' if init == 'pooled' else f'meta_{init}'
    return OLD / 'checkpoints' / f'refit_{stem}_s{seed}.pt'

def new(init, seed):
    model = engine._new_model(SPEC, seed, DEVICE)
    value = base._torch_load(checkpoint(init, seed), DEVICE)
    assert value['complete']
    model.load_state_dict(value['state_dict'])
    return model

def files():
    result = [Path(__file__), *(FROZEN / 'scripts' / name for name in
        ['research_model_engine.py', 'final_model_engine.py', 'load_research_inputs.py']),
        OLD / 'effective_config.json', OLD / 'frozen_adaptation_settings.json',
        FROZEN / 'data/frozen/INPUT_MANIFEST.json']
    result += [checkpoint(i, s) for i in INITS for s in SEEDS]
    return {str(p.relative_to(ROOT)): digest(p) for p in result}

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'design': 'POST HOC explanatory analysis after original 2025 outcomes were inspected; no confirmatory status',
      'estimand': 'Effects of identical target SGD conditional on four saved donor-trained initializations; not general algorithm superiority',
      'initializations': INITS, 'seeds': SEEDS, 'budgets': BUDGETS,
      'cohort': json.loads((FROZEN/'data/frozen/protocol.json').read_text())['targets'],
      'queries': 'Exact original common horizon1/2/3 examples in secondary2024 and test2025; no filtering by error',
      'support': 'Exact frozen canonical_frame/select_budget; latest B label dates <=2023-12-31, exactly3B examples',
      'shared_sgd': {'lrs': LRS, 'trajectory_steps': [0] + STEPS, 'batch_size':32, 'grad_clip':1.0,
                     'weight_decay':0.0, 'sampler':'CitySampler(seed+991), same sample stream for all initializations and rates'},
      'original_grid_endpoints': [{'lr':.01,'steps':3},{'lr':.03,'steps':10},{'lr':.01,'steps':30},{'lr':.03,'steps':30}],
      'original_arm_replay': 'pooled .01x3; MAML/FOMAML/Reptile .03x10; also pooled zero; original CPU/MPS prediction tolerance0.01AQI max abs',
      'diagnostics': ['support residual MSE, whole support', 'parameter L2 and relative L2 drift, body/head',
                      'per-date future absolute/squared error', 'macro city MAE/RMSE by seed/horizon/period/budget',
                      'paired adaptation-minus-zero and meta-minus-pooled at identical schedule'],
      'donor_retention': 'Fixed seven common2023 label dates per each23donor cities, ranked SHA256(city|date); all3horizons; descriptive retention on source-refit training data, not donor generalization',
      'interpretation': 'No target-based winner, step selection or tuning; observed support/query divergence and drift are descriptive, not causal proof of source mismatch',
      'runtime': {'device':'cpu','torch_threads':2,'torch_interop_threads':1},
      'source_sha256':files(), 'frozen_at_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    path = OUT/'PROTOCOL.json'
    if path.exists():
        prior=json.loads(path.read_text()); assert prior['source_sha256']==protocol['source_sha256'], 'Frozen source changed'
    else: dump(path, protocol)
    print('PROTOCOL FROZEN', digest(path), flush=True)

def loss(model, arr):
    model.eval()
    with torch.no_grad(): return float(F.mse_loss(model(arr.x), arr.residual))

def drift(model, initial):
    result={}
    for label, names in [('all',list(initial)), ('body',[n for n in initial if not n.startswith('head.')]),
                         ('head',[n for n in initial if n.startswith('head.')])]:
        d=sum(float((model.state_dict()[n]-initial[n]).double().square().sum()) for n in names)
        norm=sum(float(initial[n].double().square().sum()) for n in names)
        result[f'{label}_l2_change']=d**.5
        result[f'{label}_relative_l2_change']=(d/norm)**.5 if norm else None
    return result

def smoke():
    frame=base.canonical_frame(base._synthetic_frame('synthetic','2023-01-01','2023-03-01',1))
    support=base.select_budget(frame,30); arr=base._arrays(support,DEVICE)
    config=engine.effective_config({'device':'cpu','compute_budget_seconds':1e9})
    rows=[]
    for lr in LRS:
        original=engine._new_model(SPEC,17,DEVICE)
        model=copy.deepcopy(original); sampler=base.CitySampler(support.city.to_numpy(str),1008)
        opt=torch.optim.SGD(model.parameters(),lr=lr)
        for step in range(1,31):
            idx=torch.as_tensor(sampler.sample(32)); model.train(); opt.zero_grad(set_to_none=True)
            l=F.mse_loss(model(arr.x[idx]),arr.residual[idx]); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            if step in STEPS:
                reference=engine.fit_supervised(copy.deepcopy(original),support,
                    {'optimizer':'sgd','lr':lr,'steps':step,'weight_decay':0.},seed=17,c=config,device=DEVICE,adaptation=True).model
                delta=max(float((p-reference.state_dict()[n]).abs().max()) for n,p in model.state_dict().items())
                assert delta==0, delta
                rows.append({'lr':lr,'step':step,'parameter_max_abs_diff_vs_frozen_fit':delta})
    dump(OUT/'synthetic_smoke.json',{'passed':True,'checks':rows,'device':'cpu'})

def evaluate(model, query, support, arr, initial, *, city,budget,seed,init,lr,step):
    pred=base.predict_neural(model,query,DEVICE,batch_size=2048)
    block=query[['station_id','city','date','issue_date','horizon','last','y']].copy()
    block=block.rename(columns={'y':'y_true'})
    block['period']=np.where(block.date.dt.year==2024,'secondary_2024','test_2025')
    block['y_pred']=pred; block['abs_error']=np.abs(pred-block.y_true); block['squared_error']=(pred-block.y_true)**2
    for k,v in {'budget':budget,'seed':seed,'initialization':init,'lr':lr,'step':step}.items(): block[k]=v
    metrics=[]
    for (period,horizon),g in block.groupby(['period','horizon']):
        metrics.append({'city':city,'budget':budget,'seed':seed,'initialization':init,'lr':lr,'step':step,
                        'period':period,'horizon':int(horizon),'n':len(g),'mae':float(g.abs_error.mean()),
                        'rmse':float(np.sqrt(g.squared_error.mean()))})
    record={'city':city,'budget':budget,'seed':seed,'initialization':init,'lr':lr,'step':step,
            'support_mse':loss(model,arr),**drift(model,initial), 'support_n':len(support),
            'support_first_label':str(support.date.min().date()),'support_last_label':str(support.date.max().date()),
            'support_dates_sha256':engine._dates_digest(engine._label_dates(support)),
            'support_first_feature_date':str(support.feature_date_min.min().date()),
            'query_rows':len(query),'query_first_feature_date':str(query.feature_date_min.min().date()),
            'query_first_issue_date':str(query.issue_date.min().date()), 'device':'cpu'}
    dp=base.predict_neural(model,DONOR_PANEL,DEVICE,batch_size=2048)
    donor=DONOR_PANEL[['city','date','horizon','y']].rename(columns={'y':'y_true'}).copy()
    donor['y_pred']=dp
    for k,v in {'target_city':city,'budget':budget,'seed':seed,'initialization':init,'lr':lr,'step':step}.items(): donor[k]=v
    record['donor_retention_mae']=float(np.abs(dp-donor.y_true).mean())
    record['donor_retention_residual_mse']=float(np.mean(((dp-donor.y_true)/100.)**2))
    return block,metrics,record,donor

def parity(block, *, init,seed,city,budget,lr,step):
    method=('matched_gru_pooled_zero' if step==0 else 'matched_gru_finetune') if init=='pooled' else f'{init}_gru_sgd'
    applies=(init=='pooled' and (step==0 or (lr==.01 and step==3))) or (init!='pooled' and lr==.03 and step==10)
    if not applies: return None
    oldpath=OLD/'prediction_blocks'/f'{base._slug(city)}_b{budget}_s{seed}_{method}.csv.gz'
    original=pd.read_csv(oldpath,dtype={'station_id':str})
    keys=['city','station_id','date','issue_date','horizon','period']
    fresh=block.copy()
    for k in ['date','issue_date']: fresh[k]=fresh[k].dt.strftime('%Y-%m-%d')
    paired=fresh.merge(original,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==len(original)==len(block)
    assert np.array_equal(paired.y_true_new,paired.y_true_old)
    delta=np.abs(paired.y_pred_new-paired.y_pred_old)
    result={'city':city,'budget':budget,'seed':seed,'initialization':init,'method':method,'lr':lr,'step':step,
            'rows':len(block),'prediction_max_abs_diff':float(delta.max()),'prediction_mean_abs_diff':float(delta.mean()),
            'passed':bool(delta.max()<.01),'original_prediction_sha256':digest(oldpath)}
    assert result['passed'],result
    return result

def run():
    global DONOR_PANEL
    protocol=json.loads((OUT/'PROTOCOL.json').read_text()); assert protocol['source_sha256']==files()
    assert Path(base.__file__).resolve()==(FROZEN/'scripts/final_model_engine.py').resolve()
    assert engine.base is base
    smoke()
    bundle,_,_=load_inputs(FROZEN/'data/frozen',OUT/'input_copy')
    config=engine.effective_config(json.loads((OLD/'effective_config.json').read_text()))
    config.update({'device':'cpu','compute_budget_seconds':1e9,'resume':False})
    assert config['meta_batch_size']==32 and config['grad_clip']==1.
    data=base._bundle(bundle,config); del bundle
    selected=[]
    for donor_city,group in data['donor_val'].groupby('city'):
        dates=sorted(group.date.unique(),key=lambda d:hashlib.sha256(f'{donor_city}|{pd.Timestamp(d).date()}'.encode()).hexdigest())[:7]
        selected.append(group[group.date.isin(dates)])
    DONOR_PANEL=pd.concat(selected,ignore_index=True)
    assert len(DONOR_PANEL)==23*7*3
    base._atomic_csv(OUT/'donor_retention_panel.csv',DONOR_PANEL)
    checks=[]
    for init in INITS:
        for seed in SEEDS:
            cp=base._torch_load(checkpoint(init,seed)); sk='manual_gru' if init=='pooled' else init
            state=base._torch_load(OLD/'source_states'/f'{sk}_s{seed}.pt')
            assert all(torch.equal(cp['state_dict'][n],p) for n,p in state['state_dict'].items())
            record=json.loads((OLD/'fit_records'/f'refit_{"source_manual_gru" if init=="pooled" else "meta_"+init}_s{seed}.json').read_text())
            assert record['train_last_label']<='2023-12-31' and not record['target_validation_used']
            assert not set(record['train_city_names']) & set(data['targets'])
            checks.append({'initialization':init,'seed':seed,'checkpoint_sha256':digest(checkpoint(init,seed)),
                           'source_state_matches':True,'train_last_label':record['train_last_label'],
                           'train_cities':record['train_city_names'],'target_overlap':[]})
    dump(OUT/'checkpoint_chronology_checks.json',{'passed':True,'checks':checks})
    blocks=OUT/'blocks'; blocks.mkdir(exist_ok=True); completed=0; started=time.time()
    for city,parts in sorted(data['targets'].items()):
        query=pd.concat([parts['validation'],parts['test']],ignore_index=True)
        assert query.date.dt.year.isin([2024,2025]).all() and (query.issue_date>='2024-01-01').all()
        for budget in BUDGETS:
            support=base.select_budget(parts['adapt'],budget); arr=base._arrays(support,DEVICE)
            assert support.date.max()<=pd.Timestamp('2023-12-31')
            for seed in SEEDS:
                for init in INITS:
                    name=f'{base._slug(city)}_b{budget}_s{seed}_{init}'
                    receipt=blocks/f'{name}.json'; path=blocks/f'{name}.csv.gz'
                    if receipt.exists() and path.exists():
                        old=json.loads(receipt.read_text()); assert old['prediction_sha256']==digest(path)
                        assert old['protocol_sha256']==digest(OUT/'PROTOCOL.json')
                        assert old['donor_prediction_sha256']==digest(blocks/f'{name}_donor.csv.gz')
                        completed+=1; continue
                    initial_model=new(init,seed); initial=base._state(initial_model)
                    predictions=[]; donor_predictions=[]; metrics=[]; records=[]; replays=[]
                    b,m,r,d=evaluate(initial_model,query,support,arr,initial,city=city,budget=budget,seed=seed,init=init,lr=0.,step=0)
                    predictions.append(b); donor_predictions.append(d); metrics+=m; records.append(r)
                    p=parity(b,init=init,seed=seed,city=city,budget=budget,lr=0.,step=0)
                    if p: replays.append(p)
                    for lr in LRS:
                        model=copy.deepcopy(initial_model); opt=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=0.)
                        sampler=base.CitySampler(support.city.to_numpy(str),seed+991)
                        for step in range(1,31):
                            idx=torch.as_tensor(sampler.sample(32)); model.train(); opt.zero_grad(set_to_none=True)
                            l=F.mse_loss(model(arr.x[idx]),arr.residual[idx]); assert torch.isfinite(l)
                            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                            if step in STEPS:
                                b,m,r,d=evaluate(model,query,support,arr,initial,city=city,budget=budget,seed=seed,init=init,lr=lr,step=step)
                                predictions.append(b); donor_predictions.append(d); metrics+=m; records.append(r)
                                p=parity(b,init=init,seed=seed,city=city,budget=budget,lr=lr,step=step)
                                if p: replays.append(p)
                    base._atomic_csv(path,pd.concat(predictions,ignore_index=True))
                    donor_path=blocks/f'{name}_donor.csv.gz'
                    base._atomic_csv(donor_path,pd.concat(donor_predictions,ignore_index=True))
                    dump(receipt,{'donor_prediction_sha256':digest(donor_path),'metrics':metrics,'support_and_drift':records,'original_replays':replays,
                                  'prediction_sha256':digest(path),'protocol_sha256':digest(OUT/'PROTOCOL.json')})
                    completed+=1
                    dump(OUT/'progress.json',{'stage':'running','completed':completed,'total':216,'last':name,
                                              'elapsed_seconds':time.time()-started,'pid':os.getpid()})
                    print(completed,'/216',name,round(time.time()-started,1),flush=True)
    summarize()
    dump(OUT/'progress.json',{'stage':'complete','completed':completed,'total':216,'elapsed_seconds':time.time()-started})

def summarize():
    receipts=sorted((OUT/'blocks').glob('*.json')); assert len(receipts)==216,len(receipts)
    metrics=pd.DataFrame([r for p in receipts for r in json.loads(p.read_text())['metrics']])
    traces=pd.DataFrame([r for p in receipts for r in json.loads(p.read_text())['support_and_drift']])
    replay=pd.DataFrame([r for p in receipts for r in json.loads(p.read_text())['original_replays']])
    assert len(replay)==270 and replay.passed.all(),len(replay)
    base._atomic_csv(OUT/'city_seed_metrics.csv',metrics)
    base._atomic_csv(OUT/'support_parameter_trajectories.csv',traces)
    base._atomic_csv(OUT/'original_reproduction.csv',replay)
    zero=metrics[metrics.step==0].drop(columns=['lr','step']).rename(columns={'mae':'zero_mae','rmse':'zero_rmse'})
    paired=metrics[metrics.step>0].merge(zero,on=['city','budget','seed','initialization','period','horizon','n'],validate='many_to_one')
    paired['mae_change_from_zero']=paired.mae-paired.zero_mae
    paired['rmse_change_from_zero']=paired.rmse-paired.zero_rmse
    base._atomic_csv(OUT/'paired_adaptation_effects.csv',paired)
    pool=metrics[metrics.initialization=='pooled'].drop(columns='initialization').rename(columns={'mae':'pooled_mae','rmse':'pooled_rmse'})
    matched=metrics[metrics.initialization!='pooled'].merge(pool,on=['city','budget','seed','lr','step','period','horizon','n'],validate='many_to_one')
    matched['mae_minus_pooled']=matched.mae-matched.pooled_mae
    base._atomic_csv(OUT/'paired_meta_minus_pooled.csv',matched)
    keys=['initialization','lr','step','budget','period','horizon']
    macro=metrics.groupby(keys,as_index=False).agg(mae=('mae','mean'),rmse_city_macro=('rmse','mean'),city_seed_cases=('mae','size'))
    base._atomic_csv(OUT/'macro_metrics.csv',macro)
    harm=paired.groupby(keys,as_index=False).agg(mae_change_from_zero=('mae_change_from_zero','mean'),
         harmed_city_seed_cases=('mae_change_from_zero',lambda x:int((x>0).sum())),cases=('mae_change_from_zero','size'))
    base._atomic_csv(OUT/'macro_adaptation_effects.csv',harm)
    # Aggregate support/query association descriptively; do not use it to choose settings.
    z=traces[traces.step==0][['city','budget','seed','initialization','support_mse']].rename(columns={'support_mse':'zero_support_mse'})
    t=traces.merge(z,on=['city','budget','seed','initialization'],validate='many_to_one')
    t['support_mse_change']=t.support_mse-t.zero_support_mse
    diagnostic=paired.merge(t,on=['city','budget','seed','initialization','lr','step'],validate='many_to_one')
    base._atomic_csv(OUT/'support_query_diagnostics.csv',diagnostic)
    diag=diagnostic.groupby(keys,as_index=False).agg(support_mse_change=('support_mse_change','mean'),
          support_improved_query_harmed=('mae_change_from_zero',lambda x:0))
    # Explicit conjunction rather than marginal counts.
    diagnostic['support_improved_query_harmed']=(diagnostic.support_mse_change<0)&(diagnostic.mae_change_from_zero>0)
    diag=diagnostic.groupby(keys,as_index=False).agg(support_mse_change=('support_mse_change','mean'),
           support_improved_query_harmed=('support_improved_query_harmed','sum'),cases=('mae_change_from_zero','size'),
           relative_parameter_drift=('all_relative_l2_change','mean'))
    base._atomic_csv(OUT/'macro_support_query_diagnostics.csv',diag)
    dump(OUT/'SUMMARY.json',{'post_hoc':True,'completed_blocks':len(receipts),'evaluated_initialization_states':1944,
        'metric_rows':len(metrics),'trajectory_rows':len(traces),'original_replays':len(replay),
        'original_prediction_max_abs_diff':float(replay.prediction_max_abs_diff.max()),
        'protocol_sha256':digest(OUT/'PROTOCOL.json'), 'interpretation':'All schedules retained; no selected winner or step. Conditional saved-checkpoint diagnostic, no new donor training.',
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':2}})

def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--summarize',action='store_true')
    args=p.parse_args()
    if args.freeze: freeze()
    if args.run: run()
    if args.summarize: summarize()

if __name__=='__main__': main()
