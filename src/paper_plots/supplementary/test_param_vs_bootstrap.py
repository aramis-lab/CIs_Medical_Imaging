import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import seaborn as sns
from .test_basic_classif import format_p

from ..plot_utils import metric_labels, stat_labels, method_labels,upload_to_overleaf


def get_data(task):
    res=[]
    main_methods=['basic', 'bca', 'percentile']
    if task =='segm': 
        metrics = ["dsc", "iou", "boundary_iou", "nsd", "cldice", 'assd','masd','hd','hd_perc']
        methods=main_methods +['param_z', 'param_t']
    else:
        metrics= ['accuracy']
        methods=main_methods +["wilson","agresti_coull" ,"wald",'exact']
    path = f'../../../../results_metrics_{task}'
    
    for metric in metrics:
        if task=='segm':
            path_segm=os.path.join(path,f'aggregated_results_{metric}_mean.csv')
        else:
            path_segm=os.path.join(path,f'aggregated_results_{metric}.csv')
        df_metric = pd.read_csv(path_segm)
        for method in methods:
            values=df_metric[df_metric['n']==10][f'contains_true_stat_{method}'].values
            res.append({'metric':metric, 
                        'method':method, 
                        'coverage_values':values})
    
    return(pd.DataFrame(res))



def perform_pairwise_tests_param_boot(df_results_cov,task):
    
    
    bootstrap_methods=['basic', 'bca', 'percentile']
    
    
    if task=='segm':
        metrics=df_results_cov['metric'].unique()
 
    else:
        metrics=['accuracy']
    n_values = df_results_cov['n'].unique()
    p_values =  {str(n) : {m : {m2: None for m2 in bootstrap_methods} for m in metrics} for n in n_values}
    for n in n_values:
        print(n)
        for metric in metrics:

            for method2 in bootstrap_methods:
                if task=='segm':
                    df_boot = df_results_cov[(df_results_cov["method"]==method2) & (df_results_cov['stat']=='mean')& (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]

                    df_param= df_results_cov[(df_results_cov["method"]=='param_t')& (df_results_cov['stat']=='mean')& (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]
                else:
                    df_param= df_results_cov[(df_results_cov["method"]=='wilson')& (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]
                    df_boot = df_results_cov[(df_results_cov["method"]==method2) & (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]
                grp1 = (
                        df_param
                        .groupby(['task', 'algo'])['value']
                        .mean()
                        .reset_index(name='beta1')
                    )
             
                grp2 = (
                    df_boot
                    .groupby(['task', 'algo'])['value']
                    .mean()
                    .reset_index(name='beta2')
                )

                merged = pd.merge(grp1, grp2, on=['task', 'algo'], how='inner')

                merged = merged.dropna(subset=['beta1', 'beta2'])
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
                p_values[str(n)][metric][method2] = pval
    return p_values

def get_pvalues_param_boot(pvalues):
    pval_list = []
    locations = []

    for n, metric_dict in pvalues.items():
        for metric, method_dict in metric_dict.items():
            for method, p_val in method_dict.items():
                if p_val is not None:
                    pval_list.append(p_val)
                    locations.append(
                        (n, metric, method)
                    )

    pval_array = np.asarray(pval_list)
    return(pval_array, locations)

def reconstruct_param_boot(qvals, locations,p_vals,alphas):
    significance = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in p_vals.items()
    }
    qvalues = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in p_vals.items()
    }
    # Fill significance levels using q-values
    for (n, metric, method), q in zip(locations, qvals):
        significance[n][metric][method]= np.sum(q < alphas)
        qvalues[n][metric][method]= q

    # Fill missing values
    for n, metric_dict in p_vals.items():
        for metric, method_dict in metric_dict.items():
            for method, p_val in method_dict.items():

                if p_val is None:
                    significance[n][metric][method] = 0

    return qvalues,significance

def tell_significance(
    p_vals,
    alphas=np.array([0.001, 0.01, 0.05])
):
    pval_list = []
    locations = []

    for metric, boot_dict in p_vals.items():
        for method, param_dict in boot_dict.items():
            for param, p_val in param_dict.items():
                if p_val is not None:
                    pval_list.append(p_val)
                    locations.append(
                        (metric, boot, param)
                    )

    pval_array = np.asarray(pval_list)

    _, qvals, _, _ = multipletests(
        pval_array,
        method="fdr_bh"
    )

    significance = {
        metric: {
            boot: {
                param: None
                for param in boot_dict
            }
            for boot, boot_dict in metric_dict.items()
        }
        for metric, metric_dict in p_vals.items()
    }

    # Fill significance levels using q-values
    for (metric, boot, param), q in zip(locations, qvals):
        significance[metric][boot][param]= np.sum(q < alphas)

    # Fill missing values
    for metric, metric_dict in p_vals.items():
        for boot, boot_dict in metric_dict.items():
            for param, param_dict in boot_dict.items():

                if p_val is None:
                    significance[metric][boot][param] = 0

    return significance


def plot_significance_matrix_param_boot(significance,p_values, task,type,to_overleaf=False):
    plt.rcdefaults()
    n_values=significance.keys()
    metrics= list(next(iter(significance.values())).keys())    
  
    methods = list(next(iter(next(iter(significance.values())).values())).keys())    
    if task=='segm':
        fig, axes = plt.subplots(1,len(n_values), figsize=(15, 10), sharey=True)

    else:

        fig, axes = plt.subplots(len(n_values), 1, figsize=(12, 20), sharex=True)
    for i,(n, sign_n) in enumerate(significance.items()):
        ax=axes[i]
        global_matrix = np.zeros((len(metrics), len(methods)))
        pval_matrix = []
        p_values_n=p_values[n]
        for j, metric in enumerate(metrics):
            pval_row=[]
            metric_significance = sign_n.get(metric, {})
            p_values_metric = p_values_n.get(metric, {})
            for k, method in enumerate(methods):

                val = metric_significance.get(method)
                
                p_val = p_values_metric.get(method)

                if val is None:
                    global_matrix[j,k]=(-1)      # N/A
                elif val == 0:
                    global_matrix[j,k]=(0)       # Not significant
                else:
                    global_matrix[j,k]=(1)

                if p_val is None:
                    pval_row.append("")
               
                else:
                    pval_row.append(format_p(p_val))

            pval_matrix.append(pval_row)
            # Create p_val matrix for heatap 

        values = np.unique(global_matrix)

        # full mapping dictionary
        color_map_dict = {
        -1: "#F5F5F5",      
            0: "#F8CC80FF",     
            1: "#D55E00"
        
        }
        # extract only the colors for values that appear
        colors = [color_map_dict[v] for v in values]

        # build colormap
        cmap = ListedColormap(colors)
        
        # Plot heatmap
        labels_x = [metric_labels.get(m, m) for m in metrics]
        labels_y = [method_labels.get(m, m) for m in methods]
        sns.heatmap(
            global_matrix,
            xticklabels=labels_y,
            yticklabels=labels_x if task=='segm' else [f"n={int(float(n))}"],
            annot=pval_matrix,
            cmap=cmap,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1,
            fmt='',
            annot_kws={"fontsize": 16 if task=='classif' else 6}
        )
        ax.tick_params(axis='x', rotation=45, labelsize=14)

        ax.tick_params(axis='y', rotation=45 if task=='classif' else 0 , labelsize=16 if task=='classif' else 14)
        ax.set_title(f'n={int(float(n))}' if task=='segm' else '', fontsize=16)
        legend_elements = [
                mpatches.Patch(facecolor="#D55E00",
                            edgecolor='k',
                            label="Significant \n(FDR-adjusted p < 0.05)"),
                mpatches.Patch(facecolor="#F8CC80FF",
                            edgecolor='k',
                            label="Not significant"),
                mpatches.Patch(facecolor="#F5F5F5",
                            edgecolor='k',
                            label="Not available"),
            ]
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=12,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=12
    )
    # plt.tight_layout()
    if task=='segm':
        fig.subplots_adjust(
    left=0.1,   # more space for y tick labels
    right=0.8,  # space for legend
    # top=1,
    # bottom=0,
    wspace=0.03,
    # hspace=0.01
    )
    else:

        fig.subplots_adjust(
        left=0.08,   # more space for y tick labels
        # right=0.75,  # space for legend
        # top=1,
        # bottom=0,
        # wspace=0.07,
        # hspace=0.01
        )
    output_path= f'../clean_figs/supplementary/test_results/{type}_param_boot_{task}/all_n.pdf'
    fig.savefig(output_path)
    if to_overleaf:
        upload_to_overleaf(output_path, f"Preprint/supp_figs/Tests/{type}_param_boot_{task}.pdf", commit_msg=f"Update figure test {type}_param_boot_{task}")

    else:
        plt.show()

  
    


def main():
   for task in ['segm', 'classif']:
        print('get data')
       
        print('performing tests')
        p_values=perform_pairwise_tests_param_boot( task)
        print('significance')
        significance=tell_significance(p_values)
        print('making plot')
        plot_significance_matrix_param_boot(significance, p_values, task)

# main()

