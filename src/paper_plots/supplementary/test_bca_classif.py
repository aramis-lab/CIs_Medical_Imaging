
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
from ..plot_utils import metric_labels, stat_labels, method_labels
import scipy
import seaborn as sns

from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels, upload_to_overleaf


def perform_pairwise_tests_bca_classif(df_results):

    metrics = ['auc', 'ap']
   
    n_values=df_results['n'].unique()
    p_values = {str(n): {metric : None for metric in metrics} for n in n_values}
    for n in n_values:
    
        for metric in metrics:
                
            data_bca = df_results[(df_results["method"]=='bca') & (df_results['metric'] == metric)& (df_results['n']==n)]
            data_percentile= df_results[(df_results["method"]=='percentile') & (df_results['metric'] == metric)& (df_results['n']==n)]
            grp1 = (
                data_bca
                .groupby(['task', 'algo'])['value']
                .mean()
                .reset_index(name='beta1')
            )
        
            grp2 = (
                data_percentile
                .groupby(['task', 'algo'])['value']
                .mean()
                .reset_index(name='beta2')
            )

            merged = pd.merge(grp1, grp2, on=['task', 'algo'], how='inner')

            merged = merged.dropna(subset=['beta1', 'beta2'])
        
            if len(merged) < 2:
                pval = None
            else:
                def statistic(x, y):
                    return np.mean(x) - np.mean(y)
                
                res = permutation_test(
                    (merged['beta1'].to_numpy(), merged['beta2'].to_numpy()),
                    statistic,permutation_type='samples',
                    vectorized=False,
                    n_resamples=100000,
                    alternative='greater'
                )
                pval = res.pvalue
            p_values[str(n)][metric] = pval
            

    return p_values

