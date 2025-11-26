import math

metrics = df_segm['metric'].unique()
CI_segm_stat = df_segm[df_segm['stat'].isin(['mean', 'std'])]

n_cols = 3
n_rows = math.ceil(len(metrics) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows), sharex=False)
axes = axes.flatten()  # flatten to 1D for easy indexing

for i, metric in enumerate(metrics):
    df_all_metric = CI_segm_stat[CI_segm_stat['metric'] == metric]
    data_method = df_all_metric[df_all_metric['method'] == 'percentile']

    ax = axes[i]

    # central plot
    for stat, df_stat in data_method.groupby('stat'):
        grouped = df_stat.groupby('n')
        n_vals = grouped['coverage'].median().index.values
        medians = grouped['coverage'].median().values
        q1 = grouped['coverage'].quantile(0.25).values
        q3 = grouped['coverage'].quantile(0.75).values

        ax.plot(n_vals, medians, marker='o', label=stat_labels[stat],
                linewidth=2, markersize=8)
        ax.fill_between(n_vals, q1, q3, alpha=0.2)

    ax.set_title(f'Metric: {metric_labels[metric]}', weight='bold')
    ax.set_xlabel('Sample size', weight='bold', fontsize=14)
    ax.set_ylabel('Coverage', weight='bold', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylim(0.5, 1)
    ax.grid(True, axis='y')
    ax.legend(prop={'weight': 'bold'}, fontsize=12)

# hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("../../../../journal paper plots/segmentation/message 3 stats/spread_vs_central_all.pdf")
plt.show()
