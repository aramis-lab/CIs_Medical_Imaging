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

metric_labels = {
    'dsc': 'DSC',
    'iou': 'IoU',
    'nsd': 'NSD',
    'boundary_iou': 'Boundary IoU',
    'cldice': 'clDice',
    'assd': 'ASSD',
    'masd' : 'MASD',
    'hd': 'HD',
    'hd_perc': 'HD95',
    'balanced_accuracy': 'Balanced Accuracy',
    'ap': 'AP',
    'auc': 'AUC',
    'f1_score': 'F1 Score',
    'accuracy': 'Accuracy',
    "mcc": "MCC"
}

stat_labels = {
    'mean': 'Mean',
    'median': 'Median',
    'std': 'Standard Deviation',
    'trimmed_mean': 'Trimmed Mean',
    'iqr_length': 'IQR Length'
}

method_labels = {
    "basic": "Basic",
    "percentile": "Percentile",
    "bca": "BCa",
    "delong": "DeLong",
    "logit_transform": "Logit Transform",
    "wilson": "Wilson",
    "agresti_coull" : "Agresti-Coull",
    "exact" : "Exact \n(Clopper-Pearson)",
    "wald" : 'Wald',
    "param_t" : "Parametric t",
    "param_z" : "Parametric z"
}

method_colors = {
    "basic": "#D4461F",
    "percentile": "#8E5EE8", 
    "bca" : "#FF9742",
    "wilson" : "#DFCF3E", 
    "agresti_coull" : "#5D9336", 
    "exact" : "#DB4ADB", 
    "wald" : "#367F9C",
    "param_t" : "#999999", 
    "param_z" : "#A7C7E7"}

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
        n_values = df_results_cov['n'].unique()
 
    else:
        metrics=['accuracy']
        n_values = [10, 25]
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
                print(np.mean(merged['beta1'].to_numpy()),np.mean(merged['beta2'].to_numpy()),pval)
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
                method: {}
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in p_vals.items()
    }
    qvalues = {
        n: {
            metric: {
                method: {}
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
                param: {}
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


def plot_significance_matrix_param_boot(significance,p_values, task,type, n):
    plt.rcdefaults()
    metrics=significance.keys()
   
    methods = list(next(iter(significance.values())).keys())    
  
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    global_matrix = np.zeros((len(metrics), len(methods)))
    pval_matrix = []
    for i, metric in enumerate(metrics):
        pval_row=[]
        # Extract significance for the specific method and stat
        metric_significance = significance.get(metric, {})
        p_values_metric = p_values.get(metric, {})
        for j, method in enumerate(methods):

            val = metric_significance.get(method)
            
            p_val = p_values_metric.get(method)

            if val is None:
                global_matrix[i,j]=(-1)      # N/A
            elif val == 0:
                global_matrix[i,j]=(0)       # Not significant
            else:
                global_matrix[i,j]=(1)

            if p_val is None:
                pval_row.append("")
            elif p_val < 0.05:
                pval_row.append("<0.05")
            else:
                pval_row.append(f"{p_val:.3f}")
        pval_matrix.append(pval_row)
        # Create p_val matrix for heatap 

    values = np.unique(global_matrix)

    # full mapping dictionary
    color_map_dict = {
    -1: '#000000',
    0: '#d9d9d9',
    1: '#fdae61',
    
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
        yticklabels=labels_x,
        annot=pval_matrix,
        cmap=cmap,
        cbar=False,
        ax=ax,
        fmt='',
        annot_kws={"fontsize": 16}
    )
    ax.tick_params(axis='x', rotation=45, labelsize=14)

    ax.tick_params(axis='y', rotation=45, labelsize=14)

    ax.set_title(f"Param t vs bootstrap methods", fontsize=16)

    legend_elements = [
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5%'),
        mpatches.Patch(facecolor='#d9d9d9', edgecolor='k', label='Not significant')
    ]
    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=16
    )
    plt.tight_layout()
    
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    plt.savefig(f'../clean_figs/supplementary/test_results/{type}_param_boot_{task}/{n}.pdf')


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

