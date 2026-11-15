"""Rebuild paper figures and tables from retained, audited metric artifacts."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-root', type=Path, help='Directory containing expanded/ and reference/, or the original final_study_20261114 directory')
args = parser.parse_args()
RUN = args.results_root or HERE.parents[2] / 'results/final_study_20261114'
if (RUN/'expanded').is_dir() and (RUN/'reference').is_dir():
    EXP = RUN/'expanded'
    COM = RUN/'reference'
    if (EXP/'evaluation').is_dir(): EXP = EXP/'evaluation'
    if (COM/'evaluation').is_dir(): COM = COM/'evaluation'
else:
    EXP = RUN / 'mac_research/afml_research/evaluation'
    COM = RUN / 'mac_models/afml_final/evaluation'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})
m = pd.read_csv(EXP/'macro_metrics.csv')
c = pd.read_csv(COM/'macro_metrics.csv')
q = pd.read_csv(EXP/'paired_comparisons.csv')
sel = lambda x: x[(x.period=='test_2025') & (x.budget==90)]
fig, ax = plt.subplots(figsize=(7.1,2.5), layout='constrained')
for data,method,label,color,style in [(m,'persistence','Persistence','#656565','--'),
 (m,'mean7','Seven-day mean','#aa8b59',':'),
 (c,'transformer_pooled_zero','Compact Transformer, pooled','#156b81','-'),
 (m,'deep_transformer_pooled_zero','Deep Transformer, pooled','#bf673b','-'),
 (m,'matched_gru_finetune','Matched GRU, fine-tuned','#497f40','-')]:
 d=sel(data);d=d[d.method==method].sort_values('horizon')
 ax.plot(d.horizon,d.mae,marker='o',lw=1.6,label=label,color=color,ls=style)
ax.set(xticks=[1,2,3],xlabel='Forecast horizon (days)',ylabel='Macro MAE (AQI points)')
ax.grid(axis='y',alpha=.2);ax.legend(ncol=2,loc='upper left')
fig.savefig(HERE/'figures/horizon_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(7.1,2.25),sharey=True,layout='constrained')
for ax,h in zip(axs,[1,2,3]):
 for method,label,color,offset in [('maml_gru_sgd','MAML','#156b81',-.8),
      ('fomaml_gru_sgd','FOMAML','#497f40',0),('reptile_gru_sgd','Reptile','#bf673b',.8)]:
  d=q[(q.period=='test_2025')&(q.horizon==h)&(q.method==method)&(q.reference=='matched_gru_finetune')].sort_values('budget')
  ax.errorbar([0+offset/8,1+offset/8,2+offset/8],d.delta_mae,
   yerr=[d.delta_mae-d.delta_mae_ci_low,d.delta_mae_ci_high-d.delta_mae],
   fmt='o-',ms=3,lw=1,capsize=2,color=color,label=label)
 ax.axhline(0,color='#555555',ls='--',lw=.8)
 ax.set(title=f'{h}-day horizon',xticks=[0,1,2],xticklabels=['30','90','180'],xlabel='Fitting-label budget')
 ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('MAE minus ordinary transfer')
axs[0].legend(loc='upper right')
fig.savefig(HERE/'figures/meta_comparison.pdf');plt.close(fig)

names={'persistence':'Persistence','seasonal7':'Seasonal lag','mean7':'Seven-day mean',
 'ridge_local':'Local Ridge','histgb_local':'Local histogram boosting','randomforest_local':'Local random forest',
 'deep_gru_local':'Deep GRU, local','deep_gru_pooled_zero':'Deep GRU, pooled','deep_gru_finetune':'Deep GRU, fine-tuned',
 'deep_lstm_pooled_zero':'Deep attention LSTM, pooled','deep_tcn_pooled_zero':'Deep TCN, pooled',
 'deep_transformer_pooled_zero':'Deep Transformer, pooled','deep_mlp_pooled_zero':'Deep MLP, pooled',
 'matched_gru_local':'Matched GRU, local','matched_gru_pooled_zero':'Matched GRU, pooled',
 'matched_gru_finetune':'Matched GRU, fine-tuned','maml_gru_sgd':'MAML + SGD',
 'fomaml_gru_sgd':'FOMAML + SGD','reptile_gru_sgd':'Reptile + SGD',
 'diversity6_maml_sgd':'MAML + SGD, six donors'}
rows=[]
for method,name in names.items():
 d=sel(m);d=d[d.method==method].sort_values('horizon')
 vals=[f'{r.mae:.2f} / {r.rmse:.2f}' for r in d.itertuples()]
 rows.append(name+' & '+' & '.join(vals)+r' \\')
d=sel(c);d=d[d.method=='transformer_pooled_zero'].sort_values('horizon')
rows.append('Compact Transformer, pooled & '+' & '.join(f'{r.mae:.2f} / {r.rmse:.2f}' for r in d.itertuples())+r' \\')
(HERE/'table_results.tex').write_text(r'\begin{tabular}{lrrr}\toprule'+'\n'+r'Method & One day & Two days & Three days\\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n')
pq=q[(q.period=='test_2025')&(q.budget==90)&(q.reference=='matched_gru_finetune')&q.method.isin(['maml_gru_sgd','fomaml_gru_sgd','reptile_gru_sgd'])]
(HERE/'numeric_evidence.json').write_text(json.dumps({'main_B90':sel(m).to_dict('records'),'meta_B90':pq.to_dict('records'),
 'source_metrics':str(EXP/'macro_metrics.csv'),'comparisons':str(EXP/'paired_comparisons.csv')},indent=2))
print('Built two PDF figures, table and numerical evidence.')
