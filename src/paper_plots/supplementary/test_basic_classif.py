import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
# from ..plot_utils import metric_labels, stat_labels, method_labels
# from mlxtend.evaluate import permutation_test
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels, upload_to_overleaf

import seaborn as sns

def fit_ccp(classif_path,agreg_type):
    results = []
    if agreg_type=='micro':

        metrics= ["accuracy","ap", "auc", "f1_score"]
    else:
        metrics= ["balanced_accuracy","ap", "auc", "f1_score"]
    methods=['basic', 'bca', 'percentile',"wilson","agresti_coull" ,"wald"]
    for metric in metrics:
        
        path=os.path.join(classif_path,f'aggregated_results_{metric}.csv')
        
        df_metric_stat = pd.read_csv(path)
        for task in df_metric_stat['subtask'].unique():
            df_task = df_metric_stat[df_metric_stat['subtask'] == task]
            for algo in df_task['alg_name'].unique():
                    df_algo = df_task[df_task['alg_name'] == algo]
                    for method in methods:
                        if (method in ["wilson","agresti_coull" ,"wald"]) & (metric!='accuracy'):
                            continue
                        else:
                            n_values = df_algo['n'].to_numpy()
                            coverages = df_algo[f'contains_true_stat_{method}'].to_numpy()
                            Y = 0.95 - coverages
                            X = np.vstack([1/n_values]).T
                            beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                            rel_error = np.sqrt(res[0]) / np.linalg.norm(coverages)
                            new_row = {
                                'task': task,
                                'algo': algo,
                                'metric': metric,
                                'method': method,
                                'beta2': beta2[0],
                                'R2': rel_error
                            }
                            results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return(df_fit_results)


def perform_pairwise_tests_basic_classif(df_fit_results):
    
    
    metrics = df_fit_results['metric'].unique()
    methods = ['bca', 'percentile',"wald", "exact","agresti_coull", "wilson"]
    n_values = df_fit_results['n'].unique()
    p_values = {str(n):{metric : {m : None for m in methods} for metric in metrics} for n in n_values}

    for n in n_values: 
        for metric in metrics:
            # print(metric)
            for j in ['bca', 'percentile',"wald", "exact","agresti_coull", "wilson"]:
            
                
                if (j in ["wilson","agresti_coull" ,"wald", "exact"]) & (metric!='accuracy'):
                    continue
                data_basic = df_fit_results[(df_fit_results["method"]=='basic') & (df_fit_results['metric'] == metric)& (df_fit_results['n']==n)]
                
                data_methods= df_fit_results[(df_fit_results["method"]==j) & (df_fit_results['metric'] == metric)& (df_fit_results['n']==n)]
                # print(data_basic['beta2'].mean(),data_methods['beta2'].mean())
                grp1 = (
                    data_basic
                    .groupby(['task', 'algo'])['value']
                    .mean()
                    .reset_index(name='beta1')
                )
                
                grp2 = (
                    data_methods
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
                        alternative='less'
                    )
                    # res = permutation_test(
                    #     merged['beta1'].to_numpy(), merged['beta2'].to_numpy(),
                    #     paired=True,
                    #     func=statistic,
                    #     seed=0, num_rounds=50000
                    # )

                    pval = res.pvalue
                p_values[str(n)][metric][j] = pval

    return p_values

def get_pvalues_basic_classif(pvalues):

    pvals = []
    keys = []
    for n, metric_dict in pvalues.items():
        for metric, method_dict in metric_dict.items():
            for method, pval in method_dict.items():

                if pval is not None and not np.isnan(pval):
                    pvals.append(pval)
                    keys.append( (n,metric, method))

    pvals = np.asarray(pvals)
    return(pvals, keys)

def reconstruct_basic_classif(qvals, keys,pvalues,alphas):
    q_values = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        } for n, metric_dict in pvalues.items()
    }
 

    significant = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in pvalues.items()
    }
    
    for (n, metric, method), qval in zip(keys, qvals):
        q_values[n][metric][method] = qval
        if qval is None:
            significant[n][metric][method]=qval
        else:
            significant[n][metric][method] = np.sum(qval < alphas)

    return q_values,significant


def tell_significance(p_values, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    keys = []

    for metric, method_dict in p_values.items():
        for method, pval in method_dict.items():

            if pval is not None and not np.isnan(pval):
                pvals.append(pval)
                keys.append((metric, method))

    pvals = np.asarray(pvals)

    # BH-FDR correction
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )

    # Reconstruct nested dictionaries
    q_values = {
       
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in p_values.items()
        }
 

    significant = {
      
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in p_values.items()
       
    }
    
    for (metric, method), qval in zip(keys, qvals):
        print(qval)
        if qval is None:
     
            significant[metric][method]=qval
        else:
            significant[metric][method] = np.sum(qval < alphas)

        q_values[metric][method] = qval


    return q_values,significant


def format_p(p):
    if p is None:
        return ""
    elif p < 1e-4:
        return f"{p:.1e}"      # 2.3e-06
    elif p < 1e-3:
        return f"{p:.4f}"      # 0.0007
    elif p < 0.01:
        return f"{p:.3f}"      # 0.008
    else:
        return f"{p:.2f}"  
    
        # 0.03, 0.27
def plot_significance_matrix_basic_classif(significance, p_values, task, to_overleaf=False):
    plt.rcdefaults()
    main_methods = ['bca', 'percentile']
    if task=='macro':
        metric_order = ["balanced_accuracy","ap", "auc", "f1_score"]
        methods=main_methods
    else:
        metric_order = ["accuracy","ap", "auc", "f1_score"]
        methods=main_methods+["wilson","agresti_coull" ,"wald"]
    
    # methods=list(next(iter(significance.values())).keys())
    metrics_all = metric_order
    n_values=significance.keys()
    if task=='macro':
        fig, axes = plt.subplots(1,len(n_values), figsize=(18, 12), sharey=True  )
    else:
        fig, axes = plt.subplots(1,len(n_values), figsize=(18, 12), sharey=True  )


    for i,(n, sign_n) in enumerate(significance.items()):
        ax=axes[i]
        p_values_n=p_values[n]
        global_matrix = np.zeros((len(metrics_all), len(methods)))
        
        for i, metric in enumerate(metrics_all):
        
            for j, method in enumerate(methods):

                val = sign_n.get(metric, {}).get(method, None)
    
                if val is None :
                    global_matrix[i, j] = -1      # N/A
                elif val == 0:
                    
                    global_matrix[i, j] = 0       # Not significant
                else:
                    global_matrix[i, j] = 1
                # global_matrix[i, j] = val if val is not None else -1
            

        pval_matrix = []

        for metric in metrics_all:
            pval_row = []

            for method in methods:

                p_val = p_values_n.get(metric, {}).get(method)
                if p_val is None:
                    pval_row.append("")
                else:
                    pval_row.append(format_p(p_val))
                # if p_val is None:
                #     pval_row.append("0")
                # else:
                #     pval_row.append(
                #         f"{p_val:.6f}" if p_val >= 0.05 else "<0.05"
                #     )

            pval_matrix.append(pval_row)
        
        values = np.unique(global_matrix)
        # full mapping dictionary
        color_map_dict = {
        
           -1: "#F5F5F5",      # missing
            0: "#F8CC80FF",      # non-significant
            1: "#D55E00",      # significant
        
        }
        # extract only the colors for values that appear
        colors = [color_map_dict[v] for v in values]

        # build colormap
        cmap = ListedColormap(colors)
        mabels = [method_labels.get(m, m) for m in methods]
        metlabels=[metric_labels.get(m, m) for m in metrics_all]
        # Plot heatma
        sns.heatmap(
            global_matrix,
            annot=pval_matrix,
            xticklabels=mabels,
            yticklabels=metlabels,
            cmap=cmap,
            cbar=False,
            ax=ax,
            fmt='',
            linewidths=1,
            square=True,
            linecolor="white",
            annot_kws={"fontsize": 8 if task=='macro' else 4}
        )
        if task=='macro':
            ax.tick_params(axis='x', rotation=45, labelsize=12)

            ax.tick_params(axis='y', rotation=45, labelsize=12)
        else:
            ax.tick_params(axis='x', rotation=90, labelsize=7)

            ax.tick_params(axis='y', rotation=45, labelsize=7)
        ax.set_title(f'n={int(float(n))}', fontsize=12)
        legend_elements = [
                mpatches.Patch(facecolor="#D55E00",
                            edgecolor='k',
                            label="Significant (FDR-adjusted p < 0.05)"),
                mpatches.Patch(facecolor="#F8CC80FF",
                            edgecolor='k',
                            label="Not significant"),
                mpatches.Patch(facecolor="#F5F5F5",
                            edgecolor='k',
                            label="Not available"),
            ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.01, 0.8),
        ncol=1,
        fontsize=10,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=10
    )
    if task=='macro':
        fig.subplots_adjust(
        # left=0.08,   # more space for y tick labels
        # right=0.75,  # space for legend
        # top=1,
        # bottom=0,
        wspace=0.02,
        # hspace=0.01
        )
    else:

        fig.subplots_adjust(
        # left=0.08,   # more space for y tick labels
        # right=0.75,  # space for legend
        # top=1,
        # bottom=0,
        wspace=0.03,
        # hspace=0.01
        )
    plt.rcParams["figure.dpi"] = 200
    

    
    output_path=f"../clean_figs/supplementary/test_results/cov_basic_classif_{task}/all_n.pdf"
    fig.savefig(output_path,dpi=300, bbox_inches="tight")
    if to_overleaf:
        upload_to_overleaf(output_path, f"Preprint/supp_figs/Tests/cov_basic_classif_{task}.pdf", commit_msg="Update figure test basic classif")
    else:
        plt.show()



def main(agreg_type):
    if agreg_type=='micro':
        path='../../../../results_metrics_classif'
    else:
        path='../../../../results_metrics_classif_macro'
    print('fitting ccp')
    df_fit_results=fit_ccp(path,agreg_type)
    valid_fits=df_fit_results[df_fit_results['R2']<=0.2]
    print('performing tests')
    p_values=perform_pairwise_tests_basic_classif(valid_fits)
    print('significance')
    q_values,significance=tell_significance(p_values)

    print('making plot')
    plot_significance_matrix_basic_classif(significance, p_values,agreg_type)

# main('micro')
# main('macro')