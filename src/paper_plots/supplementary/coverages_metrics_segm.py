metric_order =['dsc', 'iou', 'nsd', 'boundary_iou', 'cldice', 'assd', 'masd', 'hd', 'hd_perc' ]

palette = sns.color_palette("colorblind", len(metric_order))
color_dict = dict(zip(metric_order, palette))
color_dict.update({
    "iou": (31/255, 119/255, 180/255),        # #1f77b4 -> RGB normalized
    "boundary_iou": (74/255, 144/255, 226/255),  # #4a90e2 -> RGB normalized
    "cldice": (1/255, 104/255, 4/255)         # #016804 -> RGB normalized
})
stats=["mean", "median", "trimmed_mean", "std", "iqr_length"]
methods = ['basic', 'percentile', 'bca', 'param_z', 'param_t']

fig, axs = plt.subplots(len(stats), len(methods), figsize=(12*len(methods), 10*len(stats)))
for i, stat in enumerate(stats):
    for j, method in enumerate(methods):
        if (stat != 'mean') and (method in ['param_z', 'param_t']):
            axs[i, j].axis('off')
            continue
        if len(stats) == 1 and len(methods) == 1:
            ax = axs
        elif len(stats) == 1 or len(methods) == 1:
            ax = axs[max(i,j)]
        else:
            ax = axs[i, j]
        data_stat=df_segm[df_segm['stat']==stat]
        data_method=data_stat[data_stat['method']==method]

        for metric, df_metric in data_method.groupby('metric'):
            medians = data_method[data_method['metric']==metric].groupby('n')['coverage'].median().values
            q1 = data_method[data_method['metric']==metric].groupby('n')['coverage'].quantile(0.25).values
            q3 = data_method[data_method['metric']==metric].groupby('n')['coverage'].quantile(0.75).values
            ax.plot(df_metric['n'].unique(), medians, marker='o', label=metric_labels[metric], color=color_dict[metric], linewidth=2, markersize=8)
            ax.fill_between(df_metric['n'].unique(), q1, q3, alpha=0.2, color=color_dict[metric])

        ax.set_title(f'Stat : {stat_labels[stat]}, Method : {method_labels[method]}', weight='bold', fontsize=14)
        ax.set_xlabel('Sample size', weight='bold', fontsize=13)
        ax.set_ylabel('Coverage', weight='bold',fontsize=13)
        ax.grid(True, axis='y')
        # Sort the legend by median value at n=125
        legend_order = (
            data_method[data_method['n'] == 125]
            .groupby('metric')['coverage']
            .median()
            .sort_values(ascending=False)
            .index
        )
        legend_order = pd.Index([metric_labels[m] for m in legend_order])

        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.get_loc(x[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        ax.legend(sorted_handles, sorted_labels, fontsize=20, loc="lower right")
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0.49,1)
        ax.set_yticks(np.arange(0.5, 1.05, 0.05))
plt.savefig(f'../../../clean_figs/cov_segm_metrics.pdf')
plt.show()