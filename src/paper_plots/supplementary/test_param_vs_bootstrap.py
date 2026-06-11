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



def perform_pairwise_tests(data, task):
    
    bootstrap_methods=['basic', 'bca', 'percentile']
    
    param_methods=[method for method in data['method'].unique() if method not in bootstrap_methods]
    metrics=data['metric'].unique()
    
    p_values =  {metric : {m : {m2: None for m2 in param_methods} for m in bootstrap_methods} for metric in metrics}
    for metric in metrics:
        
        for method1 in param_methods:
            for method2 in bootstrap_methods:
            
                data_param = data[(data["method"]==method1) & (data['metric'] == metric)]
                data_boot = data[(data["method"]==method2) & (data['metric'] == metric)]
                
                def statistic(x, y):
                    return np.mean(x) - np.mean(y)

                res = permutation_test(
                    (data_param['coverage_values'].iloc[0], data_boot['coverage_values'].iloc[0]),
                    statistic,
                    vectorized=False,
                    n_resamples=50000,
                    alternative='greater'
                )
                pval = res.pvalue

                p_values[metric][method2][method1] = pval

    return p_values


def tell_significance(
    p_vals,
    alphas=np.array([0.001, 0.01, 0.05])
):
    pval_list = []
    locations = []

    for metric, boot_dict in p_vals.items():
        for boot, param_dict in boot_dict.items():
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


def plot_significance_matrix_segm_vs_classif(significance,p_values, task):

    plt.rcdefaults()
    metrics=significance.keys()
   
    bootstrap_methods=list(next(iter(significance.values())).keys())
    
    param_methods=list(next(iter(next(iter(significance.values())).values())).keys())
  
    fig, axes = plt.subplots(1, len(metrics), figsize=(15 * len(metrics), 12))

    
    for col, metric in enumerate(metrics):
        if task=='segm':
            ax = axes[col]
        else:
            ax=axes

        # Extract significance for the specific method and stat
        metric_significance = significance.get(metric, {})
        global_matrix = np.zeros((len(bootstrap_methods), len(param_methods)))

        for i, boot in enumerate(bootstrap_methods):
            for j, param in enumerate(param_methods):
                val = metric_significance.get(boot, {}).get(param, None)
                global_matrix[i, j] = min(3, val) if val is not None else 0

        # Create p_val matrix for heatap 
        pval_matrix = []
        for i, boot in enumerate(bootstrap_methods):
            pval_row = []
            for j, param in enumerate(param_methods):
                p_val = p_values.get(metric, {}).get(boot, {}).get(param, None)
                if p_val is not None:
                    pval_row.append(f"{p_val.round(4)}" if p_val >= 0.0001 else "<0.0001")
                else:
                    pval_row.append("0")
            pval_matrix.append(pval_row)
        
        values = np.unique(global_matrix)

        # full mapping dictionary
        color_map_dict = {
            -1: '#000000',
            0: '#d9d9d9',
            1: '#fee08b',
            2: '#fdae61',
            3: '#d73027',
        }
        # extract only the colors for values that appear
        colors = [color_map_dict[v] for v in values]

        # build colormap
        cmap = ListedColormap(colors)
        
        # Plot heatmap
        labels_x = [metric_labels.get(m, m) for m in param_methods]
        labels_y = [metric_labels.get(m, m) for m in bootstrap_methods]
        sns.heatmap(
            global_matrix,
            xticklabels=labels_x,
            yticklabels=labels_y,
            annot=pval_matrix,
            cmap=cmap,
            cbar=False,
            ax=ax,
            fmt='',
            annot_kws={"fontsize": 16}
        )
        ax.tick_params(axis='x', rotation=45, labelsize=14)

        ax.tick_params(axis='y', rotation=45, labelsize=14)

        ax.set_title(f"Metric : {metric_labels[metric]}", fontsize=16)

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='k', label='1%'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5%'),
        mpatches.Patch(facecolor='#fee08b', edgecolor='k', label='10%'),
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
    plt.savefig(f'../../../../clean_figs/supplementary/test_param_vs_boot_{task}.pdf')
    plt.show()


def main():
   for task in ['segm', 'classif']:
        print('get data')
        data=get_data(task)
    
        print('performing tests')
        p_values=perform_pairwise_tests(data, task)
        print('significance')
        significance=tell_significance(p_values)
        print('making plot')
        plot_significance_matrix_segm_vs_classif(significance, p_values, task)

main()

