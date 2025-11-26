








metrics = df_segm['metric'].unique()
# Identify all columns except 'stat' and 'median'
other_cols = [c for c in df_segm.columns if c not in ["stat", "median"]]
n_cols = 3
n_rows = math.ceil(len(metrics) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows), sharex=False)
axes = axes.flatten()  # flatten to 1D for easy indexing

for i,metric in enumerate(metrics):

    df_all_metric=df_segm[df_segm['metric']==metric]
    data_method=df_all_metric[df_all_metric['method'].isin(['bca', 'percentile']) & df_all_metric['stat'].isin(['mean', 'median'])]
    ax=axes[i]
    medians = data_method.groupby(['n', 'method', 'stat'])['coverage'].median().reset_index()
    q1 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.25).reset_index()
    q3 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.75).reset_index()
    df_plot = medians.merge(q1, on=['n', 'method', 'stat'], suffixes=('_median', '_q1')).merge(q3, on=['n', 'method', 'stat'])
    df_plot.rename(columns={'coverage': 'coverage_q3'}, inplace=True)
    
    for (method, stat), df_group in df_plot.groupby(['method', 'stat']):
        linestyle = '--' if stat == 'median' else '-'
        ax.plot(
            df_group['n'], df_group['coverage_median'],
            label=f"{method_labels[method]} ({stat_labels[stat]})",
            color=method_colors[method],
            marker='o',
            linestyle=linestyle,
            linewidth=2
        )
    
    ax.set_title(f'Metric: {metric_labels[metric]}', weight='bold')
    ax.set_xlabel('Sample size',weight='bold', fontsize=16)
    ax.set_ylabel('Coverage', weight='bold', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_ylim(None, 1.01)
    ax.grid(True, axis='y')

    ax.legend(fontsize= 20)
plt.legend(fontsize= 20)
plt.tight_layout()
plt.savefig("../../../../journal paper plots/segmentation/fail_bca_all.pdf")
plt.show()
