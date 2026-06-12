import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests

from .test_basic_classif import get_pvalues_basic_classif, reconstruct_basic_classif
from .test_basic import get_pvalues_basic, reconstruct_basic

from .test_micro_vs_macro import get_pvalues_micro_macro, reconstruct_micro_macro
from .test_param_vs_bootstrap import get_pvalues_param_boot, reconstruct_param_boot
from .test_spread_vs_central import get_pvalues_spread_central, reconstruct_spread_central

from .test_WDP_segm_classif import get_pvalues_wdp_segm_classif, reconstruct_wdp_segm_classif
# from .tests_CCP_segm_vs_classif import get_pvalues_segm_classif
# from .tests_CCP_segm import get_pvalues_segm

tests=['basic_classif', 'basic','micro_macro', 'param_boot','spread_central','wdp_segm_classif']

def tell_significance(tests, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    # pvalues_all=[]
    keys = []
    keys_test=[]
    for test in tests: 
        if test=='basic_classif':
            with open(f"../pvalues/pvalues_{test}_micro.json", "r") as f:
                pvalues1 = json.load(f)
            with open(f"../pvalues/pvalues_{test}_macro.json", "r") as f:
                pvalues2 = json.load(f)
            pvalue1,key1=globals()[f'get_pvalues_{test}'](pvalues1)
            pvalue2,key2=globals()[f'get_pvalues_{test}'](pvalues2)
        
            pvals.append({'test':test +"_"+ 'micro', 
                          'pvalues':pvalue1,
                          'Key':key1})
            
            pvals.append({'test':test+ "_"+'macro', 
                          'pvalues':pvalue2,
                          'Key':key2})
            
            # pvalues_all.append(pvalue1, pvalue2)
        elif test =='param_boot':
            with open(f"../pvalues/pvalues_{test}_segm.json", "r") as f:
                pvalues1 = json.load(f)
            with open(f"../pvalues/pvalues_{test}_classif.json", "r") as f:
                pvalues2 = json.load(f)
            pvalue1,key1=globals()[f'get_pvalues_{test}'](pvalues1)
            pvalue2,key2=globals()[f'get_pvalues_{test}'](pvalues2)
            pvals.append({'test':test + "_"+'segm', 
                          'pvalues':pvalue1, 
                          'Key':key1})
            
            pvals.append({'test':test+ "_"+'classif', 
                          'pvalues':pvalue2,
                          'Key':key2})
         
        else:
            with open(f"../pvalues/pvalues_{test}.json", "r") as f:
                pvalues = json.load(f)
            pvalue,key=globals()[f'get_pvalues_{test}'](pvalues)
            pvals.append({'test':test, 
                          'pvalues':pvalue,
                          'Key':key})
        
        keys_test.append(test)

    pvalues_df=pd.DataFrame(pvals)

    pvals = np.asarray(np.concatenate(pvalues_df['pvalues'].values).ravel())
    
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )
   
    significance=[]
    i=0
    for test in pvalues_df['test'].unique():
        pvalues_test=pvalues_df[pvalues_df['test']==test]
        
        qvalues_test=qvals[i: i+len(pvalues_test['pvalues'].iloc[0])]
        i+=len(pvalues_test['pvalues'])+1
        with open(f"../pvalues/pvalues_{test}.json", "r") as f:
                pvalues = json.load(f)
        if test in ['basic_classif_micro', 'basic_classif_macro']:
            q_vals_dict,significant_dict=globals()[f'reconstruct_basic_classif'](qvalues_test, pvalues_test['Key'].iloc[0],pvalues,alphas)
          
        elif test in ['param_boot_segm', "param_boot_classif"]:
            # print(pvalues_test['Key'].iloc[0])
            significant_dict=globals()[f'reconstruct_param_boot'](qvalues_test, pvalues_test['Key'].iloc[0],pvalues,alphas)
        elif test == 'wdp_segm_classif':
            significant_dict=globals()[f'reconstruct_{test}'](qvalues_test, pvalues_test['Key'].iloc[0],pvalues,alphas)
        else:
            q_vals_dict,significant_dict=globals()[f'reconstruct_{test}'](qvalues_test, pvalues_test['Key'].iloc[0],pvalues,alphas)

        significance.append({'test':test, 
                       'significance':significant_dict})
        

    return pd.DataFrame(significance)

# print(tell_significance(tests))



