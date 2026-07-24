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
from .test_bca import plot_significance_matrix_bca
from .test_WDP_segm_classif import plot_significance_matrix_wdp_segm_classif, get_pvalues_wdp_segm_classif

from .tests_CCP_segm import plot_significance_matrix_segm, get_pvalues_segm

from .tests_CCP_segm_vs_classif import plot_significance_matrix_segm_classif, get_pvalues_segm_classif
from .make_correction_fdr import significance

def make_plots(significance,to_overleaf):
    for test in significance['test'].unique(): 
        print(test)
        significance_test = significance[significance['test']==test]['significance'].iloc[0]
        p_values= significance[significance['test']==test]['pvalues_corrected'].iloc[0]
    
        if test in ['basic_classif_micro', 'basic_classif_macro']:
            
            if test == 'basic_classif_micro':
            
                agreg='micro'
            else:
                agreg='macro'
            plot_significance_matrix_basic_classif(significance_test,p_values, agreg,to_overleaf )
    
        elif test in ['bca', 'bca_classif']:
                    
            if test=='bca':
                task='segm'
            else: 
                task='classif'
            plot_significance_matrix_bca(significance_test,p_values,task,to_overleaf)

        elif test in ['param_boot_segm', "param_boot_classif", 'param_boot_segm_width' ]:
                 
            if test== 'param_boot_segm':
                task='segm'
                type='cov'
            elif test=='param_boot_segm_width':
                task='segm'
                type='width'

            else:
                type='cov'
                task='classif'
        
            plot_significance_matrix_param_boot(significance_test,p_values, task,type,to_overleaf)
        
        elif test in ['wdp_segm_classif', 'wdp_segm_classif_macro']:
               
            if test=='wdp_segm_classif':
                task='micro'
            else:
                task='macro'
            plot_significance_matrix_wdp_segm_classif(significance_test,p_values, task,to_overleaf)

        else:
            
            globals()[f'plot_significance_matrix_{test}'](significance_test,p_values,to_overleaf)


make_plots(significance,True)