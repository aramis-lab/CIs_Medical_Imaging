import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


from .test_basic_classif import plot_significance_matrix_basic_classif ,get_pvalues_basic_classif
from .test_basic import plot_significance_matrix_basic, get_pvalues_basic

from .test_micro_vs_macro import plot_significance_matrix_micro_macro, get_pvalues_micro_macro
from .test_param_vs_bootstrap import plot_significance_matrix_param_boot, get_pvalues_param_boot
from .test_spread_vs_central import plot_significance_matrix_spread_central, get_pvalues_spread_central

from .test_WDP_segm_classif import plot_significance_matrix_wdp_segm_classif, get_pvalues_wdp_segm_classif

from .make_correction_fdr import tell_significance

tests=['basic_classif', 'basic','micro_macro', 'param_boot','spread_central','wdp_segm_classif']
significance= tell_significance(tests)

for test in significance['test'].unique(): 
    significance_test = significance[significance['test']==test]['significance'].iloc[0]
    
    with open(f"../pvalues/pvalues_{test}.json", "r") as f:
            pvalues = json.load(f)
    if test in ['basic_classif_micro', 'basic_classif_macro']:
        if test== 'basic_classif_micro':
            agreg='micro'
        else:
            agreg='macro'
        globals()[f'plot_significance_matrix_basic_classif'](significance_test,pvalues, agreg)
          
    elif test in ['param_boot_segm', "param_boot_classif"]:
        if test== 'param_boot_segm':
            task='segm'
        else:
            task='classif'
            # print(pvalues_test['Key'].iloc[0])
        globals()[f'plot_significance_matrix_param_boot'](significance_test,pvalues, task)

    else:
        globals()[f'plot_significance_matrix_{test}'](significance_test,pvalues)
