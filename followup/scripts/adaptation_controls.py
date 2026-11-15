"""Post-hoc fixed adaptation controls; frozen original source models, never source retraining."""
from pathlib import Path
import argparse, copy, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'delivery/submission_20261115/repository'
sys.path.insert(0,str(FROZEN))
from scripts import research_model_engine as eng
from scripts import final_model_engine as base
from scripts.load_research_inputs import load_inputs
OUT=ROOT/'results/explanatory_investigation_20261115/adaptation'
ORIG=ROOT/'results/final_study_20261114/mac_research/afml_research/models'
FAMILIES={'deep_gru':('gru','deep_gru_finetune','deep_gru_pooled_zero',[0,1,3,10,30,100]),
          'matched_gru':('manual_gru','matched_gru_finetune','matched_gru_pooled_zero',[0,1,2,3,10,30])}
VARIANTS={'original':{'lr_factor':1.,'head_only':False,'anchor':0.},
          'smaller_lr':{'lr_factor':.1,'head_only':False,'anchor':0.},
          'head_only':{'lr_factor':1.,'head_only':True,'anchor':0.},
          'anchor_0p01':{'lr_factor':1.,'head_only':False,'anchor':.01},
          'anchor_0p1':{'lr_factor':1.,'head_only':False,'anchor':.1}}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def protocol():
    sources={str(p.relative_to(ROOT)):sha(p) for f in FAMILIES.values() for s in [17,43,71]
             for p in [ORIG/'source_states'/f'{f[0]}_s{s}.pt']}
    return {'status':'POST_HOC explanatory investigation; no selection or causal identification',
      'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'seasonal_work':'excluded','investigation_script_sha256':sha(__file__),
      'frozen_engine_sha256':{p.name:sha(p) for p in (FROZEN/'scripts').glob('*.py')},
      'source_states_sha256':sources,'input_manifest_sha256':sha(FROZEN/'data/frozen/INPUT_MANIFEST.json'),
      'seeds':[17,43,71],'budgets':[30,90,180],'families':FAMILIES,'variants':VARIANTS,
      'anchor_definition':'lambda * sum over ALL trainable parameters of squared distance to saved initial weights; residual MSE units; original weight decay also retained',
      'training':'Exact frozen feature array, residual MSE, CitySampler(seed+991), original batch sizes/optimizer/gradient clipping; all heads jointly trained over three horizons; no target selection, stopping, retuning or source retraining',
      'evaluation':'same six target cities and exact common date/horizon masks secondary2024/main2025; support separate; forecasts same residual+last transform; no clipping',
      'parity':'Before main controls, original pooled and adapted baseline reproduced for first target B90 seed17 both families; every case original endpoint/step0 checked against original saved blocks; max prediction absolute tolerance .002 AQI',
      'inference':'date losses paired; seed, step, horizon and budget correlated repetitions, never independent samples; root coordinator performs grouped/date-block inference',
      'seen_donor_diagnostic':'Fixed retention panel: each of23 donors, six complete2023 common label dates chosen by ascending SHA256(city|date); all3horizons. Seen-source training retention, not held-out donor transfer or generalization. Does not tune controls.',
      'resume':'Atomic per city/budget/seed/family/variant compressed blocks and JSON receipts; any interrupted trajectory reruns from source; original outputs immutable'}
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol.json'
    if not p.exists():write(p,protocol())
    frozen=json.loads(p.read_text())
    assert frozen['investigation_script_sha256']==sha(__file__), 'Script changed after protocol freeze; disclose revision before continuing'
    return frozen
def inputs():return load_inputs(FROZEN/'data/frozen',OUT/'input_snapshot')[0]
def config():
    c=json.loads((ORIG/'effective_config.json').read_text());c['_deadline']=float('inf');c['resume']=False
    return c
SETTINGS=json.loads((ORIG/'frozen_adaptation_settings.json').read_text())
def model_for(family,seed,device):
    state=base._torch_load(ORIG/'source_states'/f'{FAMILIES[family][0]}_s{seed}.pt',device)
    model=eng._new_model(state['model_spec'],seed,device);model.load_state_dict(state['state_dict']);return model

def trajectory(model,support,frames,steps,preset,variant,seed,c,device,callback):
    """Exact base fitting math, with only explicitly declared control modifications."""
    variant=VARIANTS[variant];initial={n:p.detach().clone() for n,p in model.named_parameters()}
    for n,p in model.named_parameters():p.requires_grad_(not variant['head_only'] or n.startswith('head.'))
    params=[p for p in model.parameters() if p.requires_grad]
    kw={'lr':preset['lr']*variant['lr_factor'],'weight_decay':preset.get('weight_decay',0.)}
    opt=(torch.optim.Adam if preset['optimizer']=='adam' else torch.optim.SGD)(params,**kw)
    arrays=base._arrays(support,device);sampler=base.CitySampler(support.city.to_numpy(str),seed+991)
    def evaluate(step):
        with torch.no_grad():
            moved=sum(float(((p-initial[n])**2).sum().cpu()) for n,p in model.named_parameters())
            initnorm=sum(float((p**2).sum().cpu()) for p in initial.values())
        predictions=[base.predict_neural(model,f,device,batch_size=512) for f in frames]
        callback(step,predictions,{'parameter_l2':moved**.5,'relative_parameter_l2':(moved/max(initnorm,1e-30))**.5})
    evaluate(0)
    for step in range(1,max(steps)+1):
        size=int(c['meta_batch_size'] if preset['optimizer']=='sgd' else c['source_batch_size'])
        idx=torch.as_tensor(sampler.sample(size),device=device)
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(arrays.x[idx]),arrays.residual[idx])
        if variant['anchor']:
            loss=loss+variant['anchor']*sum(((p-initial[n])**2).sum() for n,p in model.named_parameters() if p.requires_grad)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(c['grad_clip']));opt.step()
        if step in steps:evaluate(step)
    return model

def orig_frame(city,budget,seed,method):
    return pd.read_csv(ORIG/'prediction_blocks'/f'{base._slug(city)}_b{budget}_s{seed}_{method}.csv.gz',dtype={'station_id':str})
def parity_error(city,budget,seed,method,frames,predictions):
    saved=orig_frame(city,budget,seed,method)
    actual=pd.concat([base._prediction_rows(f,p,method=method,budget=budget,seed=seed,period=period)
            for f,p,period in zip(frames,predictions,['secondary_2024','test_2025'])])
    keys=['period','date','horizon'];paired=actual.merge(saved,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==len(saved)==len(actual)
    assert np.array_equal(paired.y_true_new,paired.y_true_old)
    diff=float(np.max(np.abs(paired.y_pred_new-paired.y_pred_old)))
    if diff>.002:raise AssertionError(f'original parity mismatch {city} {method} {diff}')
    return diff

def smoke(device):
    torch.set_num_threads(2)
    c=config();support=base._synthetic_frame('synthetic','2023-01-01','2023-01-08',5)
    for family in FAMILIES:
        spec={'architecture':FAMILIES[family][0],'hidden':8,'layers':2};preset={**SETTINGS[FAMILIES[family][1]],'steps':3}
        model=eng._new_model(spec,17,device);baseline=copy.deepcopy(model)
        seen=[]
        trajectory(model,support,[support],[0,1,3],preset,'original',17,c,device,lambda s,p,m:seen.append((s,p,m)))
        fit=eng.fit_supervised(baseline,support,preset,seed=17,c=c,device=device,adaptation=True)
        delta=max(float((a-b).detach().abs().max().cpu()) for a,b in zip(model.parameters(),fit.model.parameters()))
        assert delta<1e-6,(family,delta)
        assert seen[0][2]['parameter_l2']==0
        initial=eng._new_model(spec,17,device);start=base._state(initial)
        trajectory(initial,support,[support],[0,1,3],preset,'head_only',17,c,device,lambda *a:None)
        assert all(torch.equal(v,initial.state_dict()[n].cpu()) for n,v in start.items() if not n.startswith('head.'))
    write(OUT/'smoke.json',{'passed':True,'device':str(device),'synthetic_exact_fitting_parity':True,'head_backbone_frozen':True})

def actual_parity(data,c,device):
    city=sorted(data['targets'])[0];parts=data['targets'][city];support=base.select_budget(parts['adapt'],90)
    frames=[parts['validation'],parts['test']];records=[]
    for family in FAMILIES:
        _,ft,pool,_=FAMILIES[family];preset=SETTINGS[ft]
        model=model_for(family,17,device)
        zero=parity_error(city,90,17,pool,frames,[base.predict_neural(model,q,device) for q in frames])
        fitted=eng.fit_supervised(model,support,preset,seed=17,c=c,device=device,adaptation=True)
        final=parity_error(city,90,17,ft,frames,[base.predict_neural(fitted.model,q,device) for q in frames])
        records.append({'city':city,'budget':90,'seed':17,'family':family,'pooled_max_abs':zero,'adapted_max_abs':final})
    write(OUT/'original_baseline_parity.json',{'passed':True,'records':records})

def donor_panel(data):
    chunks=[]
    for city,frame in data['donor_val'].groupby('city'):
        dates=sorted(frame.date.unique(),key=lambda d:hashlib.sha256(f'{city}|{pd.Timestamp(d):%Y-%m-%d}'.encode()).hexdigest())[:6]
        selected=frame[frame.date.isin(dates)].copy()
        assert len(selected)==18
        chunks.append(selected)
    panel=pd.concat(chunks,ignore_index=True)
    assert panel.city.nunique()==23 and len(panel)==414
    panel[['city','date','horizon']].to_csv(OUT/'donor_retention_panel.csv',index=False)
    return panel

def run(device):
    freeze();c=config();torch.set_num_threads(2);smoke(device);data=inputs();actual_parity(data,c,device)
    retention=donor_panel(data)
    started=time.time();completed=0;total=6*3*3*2*len(VARIANTS)
    for city,parts in sorted(data['targets'].items()):
      for budget in [30,90,180]:
       support=base.select_budget(parts['adapt'],budget);frames=[support,parts['validation'],parts['test'],retention]
       for seed in [17,43,71]:
        for family,(_,ft,pool,steps) in FAMILIES.items():
         for variant in VARIANTS:
          stem=f'{base._slug(city)}_b{budget}_s{seed}_{family}_{variant}';dest=OUT/'prediction_blocks'/f'{stem}.csv.gz';receipt=OUT/'receipts'/f'{stem}.json'
          if dest.exists() and receipt.exists():
              previous=json.loads(receipt.read_text())
              assert previous['protocol_sha256']==sha(OUT/'protocol.json')
              assert previous['prediction_sha256']==sha(dest)
              completed+=1;continue
          began=time.time();summaries=[];chunks=[];base_predictions=None;parities=[]
          def callback(step,predictions,movement):
            nonlocal base_predictions
            if base_predictions is None:base_predictions=[p.copy() for p in predictions]
            if variant=='original' and step in [0,SETTINGS[ft]['steps']]:
                parities.append(parity_error(city,budget,seed,pool if step==0 else ft,frames[1:3],predictions[1:3]))
            for frame,pred,p0,period in zip(frames,predictions,base_predictions,['support','secondary_2024','test_2025','seen_donor_retention_2023']):
                rows=base._prediction_rows(frame,pred,method=variant,budget=budget,seed=seed,period=period)
                rows['family']=family;rows['step']=step;rows['y_initial']=p0
                rows['absolute_error']=np.abs(pred-frame.y.to_numpy());rows['squared_error']=(pred-frame.y.to_numpy())**2
                rows['absolute_error_initial']=np.abs(p0-frame.y.to_numpy());rows['correction']=pred-p0
                chunks.append(rows)
                for horizon,g in rows.groupby('horizon'):
                    summaries.append({'city':city,'budget':budget,'seed':seed,'family':family,'variant':variant,'step':step,'period':period,'horizon':int(horizon),'n':len(g),'mae':g.absolute_error.mean(),'rmse':np.sqrt(g.squared_error.mean()),'initial_mae':g.absolute_error_initial.mean(),'residual_mse':g.squared_error.mean()/10000.,'mean_correction':g.correction.mean(),'rms_correction':np.sqrt((g.correction**2).mean()),**movement})
          model=model_for(family,seed,device)
          trajectory(model,support,frames,steps,SETTINGS[ft],variant,seed,c,device,callback)
          dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.gz')
          for index,chunk in enumerate(chunks):chunk.to_csv(tmp,index=False,mode='w' if index==0 else 'a',header=index==0,compression='gzip')
          tmp.replace(dest)
          write(receipt,{'complete':True,'seconds':time.time()-began,'rows':sum(len(x) for x in chunks),'prediction_sha256':sha(dest),'protocol_sha256':sha(OUT/'protocol.json'),'original_max_abs':max(parities) if parities else None,'summaries':summaries})
          completed+=1
          write(OUT/'progress.json',{'stage':'controls','completed':completed,'total':total,'elapsed_seconds':time.time()-started,'latest':stem,'pid':os.getpid()})
          print(f'{completed}/{total} {stem} {time.time()-began:.1f}s',flush=True)
          del model,chunks
    summarize()
def summarize():
    receipts=[json.loads(p.read_text()) for p in sorted((OUT/'receipts').glob('*.json'))]
    metrics=pd.DataFrame([r for receipt in receipts for r in receipt['summaries']]);metrics.to_csv(OUT/'metrics.csv',index=False)
    keys=['city','budget','family','variant','step','period','horizon']
    mean=metrics.groupby(keys,as_index=False)[['mae','rmse','initial_mae','mean_correction','rms_correction','parameter_l2','relative_parameter_l2']].mean()
    mean['mae_delta']=mean.mae-mean.initial_mae;mean.to_csv(OUT/'seed_averaged_metrics.csv',index=False)
    macro=mean.groupby(['family','variant','step','period'],as_index=False)[['mae','initial_mae','mae_delta','rms_correction','parameter_l2','relative_parameter_l2']].mean();macro.to_csv(OUT/'macro_descriptive.csv',index=False)
    write(OUT/'completion.json',{'complete':len(receipts)==540,'blocks':len(receipts),'prediction_rows':sum(r['rows'] for r in receipts),'max_original_prediction_difference':max(r['original_max_abs'] or 0 for r in receipts),'original_parity_passed':True,'all_metrics_finite':bool(np.isfinite(metrics.select_dtypes('number')).all().all()),'note':'Descriptive aggregation; no independence assumption across seeds/steps/horizons/budgets; root performs uncertainty analysis'})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','smoke','parity','run','summarize']);parser.add_argument('--device',default='mps');args=parser.parse_args()
    device=torch.device(args.device)
    if args.command=='freeze':freeze()
    elif args.command=='smoke':freeze();smoke(device)
    elif args.command=='parity':freeze();torch.set_num_threads(2);actual_parity(inputs(),config(),device)
    elif args.command=='run':run(device)
    else:summarize()
