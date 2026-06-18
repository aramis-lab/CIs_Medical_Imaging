import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from .make_fit import df_fit_results, df_fit_results_micro, df_fit_results_macro,df_fit_results_wdp,df_fit_results_classif_wdp, segm_path, micro_path

from .test_basic_classif import perform_pairwise_tests_basic_classif
from .test_basic import perform_pairwise_tests_basic

from .test_micro_vs_macro import perform_pairwise_tests_micro_macro
from .test_param_vs_bootstrap import perform_pairwise_tests_param_boot
from .test_spread_vs_central import perform_pairwise_tests_spread_central

from .test_WDP_segm_classif import perform_pairwise_tests_wdp_segm_classif
from .tests_CCP_segm_vs_classif import perform_pairwise_tests_segm_classif
from .tests_CCP_segm import perform_pairwise_tests_segm

df_fit_results=df_fit_results[df_fit_results['R2']<=0.1]
df_fit_results_micro=df_fit_results_micro[df_fit_results_micro['R2']<=0.1]
df_fit_results_macro=df_fit_results_macro[df_fit_results_macro['R2']<=0.1]
df_fit_results_wdp=df_fit_results_wdp[df_fit_results_wdp['R2']<=0.1]
df_fit_results_classif_wdp=df_fit_results_classif_wdp[df_fit_results_classif_wdp['R2']<=0.1]

# print('testing micro macro')
# pvalues_micro_macro,i= perform_pairwise_tests_micro_macro(df_fit_results_micro, df_fit_results_macro)
# with open("../pvalues/pvalues_micro_macro.json", "w") as f:
#     json.dump(pvalues_micro_macro, f, indent=4)

# print('testing basic')
# pvalues_basic_classif_micro=perform_pairwise_tests_basic_classif(df_fit_results_micro)
# pvalues_basic_classif_macro=perform_pairwise_tests_basic_classif(df_fit_results_macro)
# pvalues_basic=perform_pairwise_tests_basic(df_fit_results)

# with open("../pvalues/pvalues_basic.json", "w") as f:
#     json.dump(pvalues_basic, f, indent=4)

# with open("../pvalues/pvalues_basic_classif_macro.json", "w") as f:
#     json.dump(pvalues_basic_classif_macro, f, indent=4)

# with open("../pvalues/pvalues_basic_classif_micro.json", "w") as f:
#     json.dump(pvalues_basic_classif_micro, f, indent=4)



# print('testing param boot')
# pvalues_param_boot_segm= perform_pairwise_tests_param_boot('segm', segm_path)
# pvalues_param_boot_classif= perform_pairwise_tests_param_boot('classif',micro_path)

# with open("../pvalues/pvalues_param_boot_classif.json", "w") as f:
#     json.dump(pvalues_param_boot_classif, f, indent=4)

# with open("../pvalues/pvalues_param_boot_segm.json", "w") as f:
#     json.dump(pvalues_param_boot_segm, f, indent=4)

# print('testing spread central')
# pvalues_spread_central= perform_pairwise_tests_spread_central(df_fit_results)
# with open("../pvalues/pvalues_spread_central.json", "w") as f:
#     json.dump(pvalues_spread_central, f, indent=4)

# print('testing wdp segm classif')
# pvalues_wdp_segm_classif= perform_pairwise_tests_wdp_segm_classif(df_fit_results_wdp, df_fit_results_classif_wdp)
# with open("../pvalues/pvalues_wdp_segm_classif.json", "w") as f:
#     json.dump(pvalues_wdp_segm_classif, f, indent=4)

print('testing segm classif')
pvalues_segm_classif_micro = perform_pairwise_tests_segm_classif(df_fit_results, df_fit_results_micro)
pvalues_segm_classif_macro = perform_pairwise_tests_segm_classif(df_fit_results, df_fit_results_macro)

with open("../pvalues/pvalues_segm_classif_micro.json", "w") as f:
    json.dump(pvalues_segm_classif_micro, f, indent=4)

with open("../pvalues/pvalues_segm_classif_macro.json", "w") as f:
    json.dump(pvalues_segm_classif_macro, f, indent=4)

# print('testing segm')
# pvalues_segm= perform_pairwise_tests_segm(df_fit_results)

# with open("../pvalues/pvalues_segm.json", "w") as f:
#     json.dump(pvalues_segm, f, indent=4)

tests=['basic_classif', 'basic','micro_macro', 'param_boot','spread_central','wdp_segm_classif','segm_classif','segm']


def tell_significance(tests, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    keys = []
    keys_test=[]
    for test in tests: 

        if test=='basic_classif':
            pvalue1,key1=globals()[f'get_pvalues_{test}'](globals()[f'pvalues_{test}_micro'])
            pvalue2,key2=globals()[f'get_pvalues_{test}'](globals()[f'pvalues_{test}_macro'])

            pvals.append({'test':test + 'micro', 
                          'pvalues':pvalue1,
                          'Key':key1})
            
            pvals.append({'test':test+ 'macro', 
                          'pvalues':pvalue2,
                          'Key':key2})
            
            # pvals.append([pvalue1,pvalue2])
            # keys.append([key1,key2])
        elif test =='param_boot':
            pvalue1,key1=globals()[f'get_pvalues_{test}'](globals()[f'pvalues_{test}_segm'])
            pvalue2,key2=globals()[f'get_pvalues_{test}'](globals()[f'pvalues_{test}_classif'])
            pvals.append({'test':test + 'segm', 
                          'pvalues':pvalue1, 
                          'Key':key1})
            
            pvals.append({'test':test+ 'classif', 
                          'pvalues':pvalue2,
                          'Key':key2})
            
            # pvals.append([pvalue1,pvalue2])
            # keys.append([key1,key2])
        else:
            pvalue,key=globals()[f'get_pvalues_{test}'](globals()[f'pvalues_{test}'])
            pvals.append({'test':test+ 'classif', 
                          'pvalues':pvalue,
                          'Key':key})
            # pvals.append(pvalue)
            # keys.append(key)

        keys_test.append(test)

    pvalues_df=pd.DataFrame(pvals)

    pvals = np.asarray(pvalues_df['pvalues'].values)

    # BH-FDR correction
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )
    pvalues_df['pvalues_corrected']=qvals
    significance=[]
    for test in tests: 
        qvalues_test=pvalues_df[pvalues_df['test']==test]
        q_vals_dict,significant_dict=globals()[f'reconstruct_{test}'](qvalues_test['pvalues'], qvalues_test['keys'].values,globals()[f'pvalues_{test}'],alphas)
        significance.append({'test':test, 
                       'significance':significant_dict})
        
    

    return significance



# significance=tell_significance(tests)

