"""Analysis of predeclared higher-capacity / second-order meta-learning arms."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.final_evaluation import evaluate,_markdown_table

def analyze_research(predictions,fit_log,outdir,protocol,compact_root=None):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    p=dict(protocol)
    assert 'expected_methods' in p, 'Supply the pre-result frozen method inventory'
    assert set(p['expected_methods'])==set(predictions.method.unique()), 'Incomplete research method inventory'
    p['plot_methods']=[m for m in ['persistence','ridge_local','histgb_local','randomforest_local',
        'deep_gru_finetune','deep_lstm_finetune','deep_tcn_finetune','deep_transformer_finetune',
        'deep_mlp_finetune','matched_gru_finetune','maml_gru_sgd','fomaml_gru_sgd','reptile_gru_sgd'] if m in p['expected_methods']]
    comparisons=[]
    for algo in ['maml','fomaml','reptile']:
        comparisons.extend([
          {'method':f'{algo}_gru_sgd','reference':'matched_gru_finetune','kind':'native_meta_minus_matched_transfer'},
          {'method':f'{algo}_gru_adam','reference':f'{algo}_gru_sgd','kind':'optimizer_adam_minus_native_sgd'}])
    comparisons.extend([
      {'method':'maml_gru_sgd','reference':'fomaml_gru_sgd','kind':'second_order_minus_first_order'},
      {'method':'maml_gru_sgd','reference':'reptile_gru_sgd','kind':'maml_minus_reptile'},
      {'method':'diversity6_pooled_zero','reference':'matched_gru_pooled_zero','kind':'six_donor_pool_minus_all_donors'},
      {'method':'diversity6_finetune','reference':'matched_gru_finetune','kind':'six_donor_pool_minus_all_donors'},
      {'method':'diversity6_maml_sgd','reference':'maml_gru_sgd','kind':'six_donor_pool_minus_all_donors'}])
    p['comparisons']=[s for s in comparisons if s['method'] in p['expected_methods'] and s['reference'] in p['expected_methods']]
    p['fit_costs']=fit_log
    summary=evaluate(predictions,out,p)
    macro=pd.read_csv(out/'macro_metrics.csv');paired=pd.read_csv(out/'paired_comparisons.csv')
    primary=macro[macro.period=='test_2025']
    rows=['# Expanded research results','',
      'This study was specified before the compact reference target scores were inspected. It expands computational effort to answer capacity/optimization, meta-gradient, adaptation-optimizer and donor-pool questions. All conclusions below are calculated from saved observed-target forecasts. The target periods did not tune any model.','',
      '## Larger models and donor-only search','',
      'Five deeper architectures use the same legal seven-day inputs and target fitting-label budgets as the compact reference. Their training and search budgets are larger. The comparison describes the combined capacity/optimization package; it cannot isolate parameter count as a causal factor. A compact model remains a serious benchmark.','',
      _markdown_table(primary.sort_values(['budget','horizon','mae']).groupby(['budget','horizon'],as_index=False).head(5),
       ['budget','horizon','method','mae','rmse','skill_mae','cities_improved_vs_persistence','n_cities'],45),'',
      'The lowest test error is descriptive after comparing many methods. It is not a new donor-selected deployment decision.','',
      '## Full second-order MAML and matched controls','',
      'Full MAML differentiates the query loss through support-gradient updates. FOMAML omits those Hessian terms; Reptile moves toward support-adapted weights. Their manual differentiable GRU backbone and native SGD adaptation are matched to ordinary transfer controls. Correctness checks include a genuine nonzero difference between first- and second-order gradients on a synthetic nonlinear task. These checks verify the implementation; target errors determine usefulness.','']
    main_pairs=paired[paired.period=='test_2025']
    for kind,title in [('native_meta_minus_matched_transfer','Meta-learning versus ordinary matched transfer'),
                       ('second_order_minus_first_order','Second-order versus first-order gradients'),
                       ('maml_minus_reptile','MAML versus Reptile'),
                       ('optimizer_adam_minus_native_sgd','Changing the final adaptation optimizer'),
                       ('six_donor_pool_minus_all_donors','Six versus the full donor pool')]:
        d=main_pairs[main_pairs.kind==kind]
        if not len(d):continue
        rows.extend([f'### {title}','',
          f'Lower macro MAE occurred in {int((d.delta_mae<0).sum())}/{len(d)} prespecified cells; '
          f'{int((d.delta_mae_ci_high<0).sum())} intervals lie entirely below zero. Negative differences favor the listed method. '
          'Intervals are unadjusted for multiple comparisons and condition on the selected six target cities.','',
          _markdown_table(d,['budget','horizon','method','reference','delta_mae','delta_mae_ci_low','delta_mae_ci_high','cities_improved','n_cities'],27),''])
    rows.extend(['The optimizer comparison includes donor-selected schedules, so its effect includes optimizer and schedule together. The six-city donor ablation also changes the amount of unique source data; matched update counts do not isolate task diversity alone. Neither comparison proves a physical pollution mechanism.',''])
    if compact_root is not None:
        root=Path(compact_root)
        old=pd.read_csv(root/'evaluation/macro_metrics.csv')
        old=old[old.period=='test_2025']
        maps={f'deep_{a}_{mode}':f'{a}_{mode}' for a in ['gru','lstm','tcn','transformer','mlp'] for mode in ['local','pooled_zero','finetune']}
        d=primary[primary.method.isin(maps)].copy();d['compact_method']=d.method.map(maps)
        c=d.merge(old[['budget','horizon','method','mae','rmse']].rename(columns={'method':'compact_method','mae':'compact_mae','rmse':'compact_rmse'}),on=['budget','horizon','compact_method'],validate='many_to_one')
        c['large_minus_compact_mae']=c.mae-c.compact_mae
        c.to_csv(out/'capacity_compute_comparison.csv',index=False)
        rows.extend(['## Comparison with the compact reference','',
          f'The larger package has lower observed macro MAE in {int((c.large_minus_compact_mae<0).sum())}/{len(c)} matched architecture/mode/budget/horizon cells. '
          'This is a descriptive cross-run comparison, conditional on the stated backend and training budgets. The independent audits verify common labels, dates and features.','',
          _markdown_table(c,['budget','horizon','method','mae','compact_method','compact_mae','large_minus_compact_mae'],30),''])
    rows.extend(['## What can be concluded','',
      'The registered hypothesis is supported only where cross-city methods improve on local training and a useful simple forecast at the stated horizon and fitting-label budget. Meta-learning adds value only where its matched comparisons show an advantage. Results may be mixed or negative. Model count and GPU time are not substitutes for these comparisons.','',
      'The source/identity/coverage limits, simulated fitting-label scarcity, recency/season confounding, frozen-2023 model aging, six-city convenience sampling, finite optimization budgets and unverified publication latency remain. Broad hyperparameter search can overfit donor validation and does not establish global optimality.','',
      'Further work: predeclare truly new target cities and fresh rolling-origin refits; match source volume in donor-diversity experiments; expand independent task diversity; compare longer histories with a common missing-input policy; validate station history and issue-time availability; test high-pollution calibration; and reserve final external confirmation before deployment.',''])
    (out/'RESEARCH_RESULTS.md').write_text('\n'.join(rows))
    return summary
