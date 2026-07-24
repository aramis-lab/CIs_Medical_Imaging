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
from .tests_CCP_segm_vs_classif import get_pvalues_segm_classif, reconstruct_segm_classif
from .tests_CCP_segm import get_pvalues_segm, reconstruct_segm
from .test_bca import get_pvalues_bca, reconstruct_bca

tests=['basic_classif', 'basic','micro_macro', 'param_boot','spread_central','wdp_segm_classif', 'segm_classif', 'segm', 'bca', 'bca_classif']

def tell_significance(tests, kde, alphas=np.array([0.001, 0.01, 0.05])):

    pvals = []
    # pvalues_all=[]
    keys = []
    keys_test=[]
    for test in tests: 
        print(test)
        if test=='basic_classif':
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_micro_by_n.json"), "r") as f:
                pvalues1 = json.load(f)
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_macro_by_n.json"), "r") as f:
                pvalues2 = json.load(f)
            pvalue1,key1=globals()[f'get_pvalues_{test}'](pvalues1)
            pvalue2,key2=globals()[f'get_pvalues_{test}'](pvalues2)
            print(len(pvalue2), len(pvalue1))

            pvals.append({'test':test +"_"+ 'micro', 
                          'pvalues':pvalue1,
                          'Key':key1})
            
            pvals.append({'test':test+ "_"+'macro', 
                          'pvalues':pvalue2,
                          'Key':key2})
            
            # pvalues_all.append(pvalue1, pvalue2)
        elif test=='wdp_segm_classif':
            with open(f"../pvalues/pvalues_segm_classif_width_by_n.json", "r") as f:
                pvalues1 = json.load(f)
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_segm_classif_macro_width_by_n.json"), "r") as f:
                pvalues = json.load(f)
        
            pvalue1,key1=globals()[f'get_pvalues_{test}'](pvalues1)
            pvalue,key=globals()[f'get_pvalues_{test}'](pvalues)
            # print(len(pvalue2), len(pvalue1))
            pvals.append({'test':test, 
                          'pvalues':pvalue1, 
                          'Key':key1})
            
            pvals.append({'test':test+ "_"+'macro', 
                          'pvalues':pvalue,
                          'Key':key})
         
        elif test in ['bca', 'bca_classif']:
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_by_n.json"), "r") as f:
                pvalues = json.load(f)
            pvalue,key=globals()[f'get_pvalues_bca'](pvalues)
            print(len(pvalue))
            pvals.append({'test':test, 
                          'pvalues':pvalue,
                          'Key':key})
         
        elif test =='param_boot':
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_segm_by_n.json"), "r") as f:
                pvalues1 = json.load(f)
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_classif_by_n.json"), "r") as f:
                pvalues2 = json.load(f)
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_segm_width_by_n.json"), "r") as f:
                pvalues3 = json.load(f)

            pvalue1,key1=globals()[f'get_pvalues_{test}'](pvalues1)
            pvalue2,key2=globals()[f'get_pvalues_{test}'](pvalues2)
            pvalue3,key3=globals()[f'get_pvalues_{test}'](pvalues3)

            pvals.append({'test':test + "_"+'segm', 
                          'pvalues':pvalue1, 
                          'Key':key1})
            
            pvals.append({'test':test+ "_"+'classif', 
                          'pvalues':pvalue2,
                          'Key':key2})
            pvals.append({'test':test+ '_segm_width', 
                          'pvalues':pvalue3,
                          'Key':key3})
        elif test=='segm':
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}.json"), "r") as f:
                pvalues = json.load(f)
            pvalue,key=globals()[f'get_pvalues_{test}'](pvalues)
           

            pvals.append({'test':test, 
                          'pvalues':pvalue,
                          'Key':key})
        # elif test =='segm_classif':
            # with open(f"../pvalues/pvalues_{test}_micro_by_n.json", "r") as f:
            #     pvalues1 = json.load(f)
            # with open(f"../pvalues/pvalues_{test}_macro_by_n.json", "r") as f:
            #     pvalues2 = json.load(f)
            # pvalue1,key1=globals()[f'get_pvalues_{test}'](pvalues1)
            # pvalue2,key2=globals()[f'get_pvalues_{test}'](pvalues2)
            # pvals.append({'test':test + "_"+'micro', 
            #               'pvalues':pvalue1, 
            #               'Key':key1})
            
            # pvals.append({'test':test+ "_"+'macro', 
            #               'pvalues':pvalue2,
            #               'Key':key2})
        else:
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_by_n.json"), "r") as f:
                pvalues = json.load(f)
            pvalue,key=globals()[f'get_pvalues_{test}'](pvalues)
            print(len(pvalue))
            pvals.append({'test':test, 
                          'pvalues':pvalue,
                          'Key':key})
        
        keys_test.append(test)

    pvalues_df=pd.DataFrame(pvals)

    pvals = np.concatenate(pvalues_df["pvalues"].to_list())
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )
   
    significance=[]
    i=0
    for _, row in pvalues_df.iterrows():
        test = row["test"]
        pvalues_test = row["pvalues"]
        keys = row["Key"]
    
        # pvalues_test=pvalues_df[pvalues_df['test']==test]
        n = len(pvalues_test)

        qvalues_test = qvals[i: i + n]
        i += n

        if test in ['wdp_segm_classif', 'wdp_segm_classif_macro']:
            if test =='wdp_segm_classif':
                with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_segm_classif_width_by_n.json"), "r") as f:
                    pvalues = json.load(f)
            else:
                with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_segm_classif_macro_width_by_n.json"), "r") as f:
                    pvalues = json.load(f)
            
        

        elif test=='segm':
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_segm.json"),"r") as f:
                pvalues = json.load(f)
        else: 
            with open(os.path.join("../results_ablations", f"{kde}/pvalues/pvalues_{test}_by_n.json"),"r") as f:
                    pvalues = json.load(f)
        if test in ['basic_classif_micro', 'basic_classif_macro']:
            q_vals_dict,significant_dict=globals()[f'reconstruct_basic_classif'](qvalues_test, keys,pvalues,alphas)
         
        elif test in ['param_boot_segm', "param_boot_classif", 'param_boot_segm_width']:
            # print(keys)
            q_vals_dict,significant_dict=globals()[f'reconstruct_param_boot'](qvalues_test, keys,pvalues,alphas)
        elif test in ['wdp_segm_classif', 'wdp_segm_classif_macro']:
            q_vals_dict,significant_dict=globals()[f'reconstruct_wdp_segm_classif'](qvalues_test, keys,pvalues,alphas)
        elif test in ['segm_classif_micro', 'segm_classif_macro']:
            q_vals_dict,significant_dict=globals()[f'reconstruct_segm_classif'](qvalues_test, keys,pvalues,alphas)
        elif test=='segm':
            q_vals_dict,significant_dict=globals()[f'reconstruct_{test}'](qvalues_test, keys,pvalues,alphas)
        elif test in ['bca', 'bca_classif']:
            q_vals_dict,significant_dict=globals()[f'reconstruct_bca'](qvalues_test, keys,pvalues,alphas)
        
                                                
        else:
            q_vals_dict,significant_dict=globals()[f'reconstruct_{test}'](qvalues_test, keys,pvalues,alphas)
        significance.append({'test':test, 
                       'significance':significant_dict, 
                       'pvalues':pvalues,
                       'pvalues_corrected':q_vals_dict})
        

    return pd.DataFrame(significance)

tests=['basic_classif', 'basic','micro_macro', 'param_boot','spread_central','wdp_segm_classif', 'segm', 'segm_classif', 'bca', 'bca_classif']


def sum_leaves(obj):
    if isinstance(obj, dict):
        
        return np.nansum([sum_leaves(v) for v in obj.values()])
    else:
        return obj if obj is not None else 0
    
def count_non_none_leaves(obj):
    if isinstance(obj, dict):
        return sum(count_non_none_leaves(v) for v in obj.values())
    else:
        return 0 if obj is None else 1
ablation_dir="../results_ablations"
results_ablation=[]
test_mapping = {
    "basic_classif_micro": "basic",
    "basic_classif_macro": "basic",
    "basic": "basic",

    "wdp_segm_classif": "wdp_segm_classif",
    "wdp_segm_classif_macro": "wdp_segm_classif",

    "bca": "bca",
    "bca_classif": "bca",
}

kde_mapping = {
    "epanechnikov_adaptive_trimmed": "epan.adapt.trimmed",
    "gaussian_adaptive": "gauss.adapt.",
    "epanechnikov_scott_trimmed": "epan.scott.trimmed",

    "gaussian_scott": "gauss.scott"
    
}
for kde in os.listdir(ablation_dir):
    if kde in ['epanechnikov_scott_old','epanechnikov_adaptive']:
        continue

    if kde.startswith("."):
        continue

    significance= tell_significance(tests, kde, alphas=[0.05])
    for test in significance['test'].unique():

        sign_test=significance[significance['test']==test]
        significance_dict=sign_test['significance'].to_numpy()[0]
        
        results_ablation.append({'kde':kde, 
                                 'test':test, 
                                 'sum of significance':sum_leaves(significance_dict), 
                                  'total_number of tests':count_non_none_leaves(significance_dict) , 
                                  'proportion of not significance': 1- sum_leaves(significance_dict)/count_non_none_leaves(significance_dict),}
                                  )
        
        
        # results_ablation.append({'kde':kde, 
        #                          'test':sign_test['test'],
        #                         'significant_dict':sign_test['significant_dict']})
    

results_ablation_df=pd.DataFrame(results_ablation)
results_ablation_df["test_grouped"] = results_ablation_df["test"].replace(test_mapping)
results_ablation_df["kde"] = results_ablation_df["kde"].replace(kde_mapping)

df_merged = (
    results_ablation_df.groupby(["kde", "test_grouped"], as_index=False)
      .agg({
          "sum of significance": "sum",
          "total_number of tests": "sum"
      })
)
df_merged["proportion of not significance"] = round(
    (df_merged["sum of significance"] /
    df_merged["total_number of tests"])*100,1)
print(df_merged)
table = df_merged.pivot(
    index="test_grouped",
    columns="kde",
    values="proportion of not significance"
)
table = table.rename(columns={"epanechnikov_adaptive_trimmed": "epanechnikov_adaptive_trimmed (ours)"})
new_order = [
    "basic",
    "bca",
    "param_boot_segm",
    "param_boot_segm_width",
    "param_boot_classif",
    "segm_classif",
    "wdp_segm_classif",

    "segm",
    "micro_macro",
    "spread_central"
]
table=table.reindex(new_order)
latex_table = table.to_latex(float_format="%.1f")
print(latex_table)
print(table)