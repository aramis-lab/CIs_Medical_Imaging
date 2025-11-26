





# Color palettes
palette = sns.color_palette("Blues", len(metric_order))
color_dict = dict(zip(metric_order, palette))
metrics_classif = ["balanced_accuracy", "ap", "auc", "f1_score"]
palette_classif = sns.color_palette("Reds", len(metrics_classif))
color_dict_classif = dict(zip(metrics_classif, palette_classif))

df_macro_perc = df_macro[(df_macro['n'] <= 250) & (df_macro['method'] == 'percentile')]
df_segm_perc = df_segm[(df_segm['n'] <= 250) & (df_segm['method'] == 'percentile') &
                       (df_segm['metric'].isin(["boundary_iou", "nsd", "iou", "dsc", "cldice"]))]
df_segm_perc_width = df_segm_width[(df_segm_width['n'] <= 250) & (df_segm_width['method'] == 'percentile') &
                                   (df_segm_width['metric'].isin(["boundary_iou", "nsd", "iou", "dsc", "cldice"]))]
df_macro_perc_width = df_macro_width[(df_macro_width['n'] <= 250) & (df_macro_width['method'] == 'percentile')]
stats = df_segm_perc['stat'].unique()
n_stats = len(stats)
fig, axs = plt.subplots(n_stats, 2, figsize=(26, 10 * n_stats))
if n_stats == 1:
    axs = np.array([axs])  # keep consistent 2D structure

for row, stat in enumerate(stats):

    # ============================================================
    # COVERAGE (left column)
    # ============================================================
    ax = axs[row, 0]

    # --- CLASSIFICATION ---
    medians = df_macro_perc.groupby(['n', 'stat'])['value'].median().reset_index()
    q1 = df_macro_perc.groupby(['n', 'stat'])['value'].quantile(0.25).reset_index()
    q3 = df_macro_perc.groupby(['n', 'stat'])['value'].quantile(0.75).reset_index()
    df_plot = (
        medians.merge(q1, on=['n', 'stat'], suffixes=('_median', '_q1'))
        .merge(q3, on=['n', 'stat'])
        .rename(columns={'value': 'value_q3'})
    )

    for stat_classif in df_plot['stat'].unique():
        df_stat = df_plot[df_plot['stat'] == stat_classif]
        ax.plot(df_stat['n'], df_stat['value_median'], marker='o',
                color=color_dict_classif[stat_classif], linewidth=2,
                label=metric_labels[stat_classif])
        ax.plot(df_stat['n'], df_stat['value_q1'], linestyle="--",
                color=color_dict_classif[stat_classif], linewidth=1)
        ax.plot(df_stat['n'], df_stat['value_q3'], linestyle="--",
                color=color_dict_classif[stat_classif], linewidth=1)
        ax.fill_between(df_stat['n'], df_stat['value_q1'], df_stat['value_q3'],
                        alpha=0.2, color=color_dict_classif[stat_classif])

    # --- SEGMENTATION ---
    df_segm_mean = df_segm_perc[df_segm_perc['stat'] == stat]
    medians = df_segm_mean.groupby(['n', 'metric'])['coverage'].median().reset_index()
    q1 = df_segm_mean.groupby(['n', 'metric'])['coverage'].quantile(0.25).reset_index()
    q3 = df_segm_mean.groupby(['n', 'metric'])['coverage'].quantile(0.75).reset_index()
    df_plot = (
        medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
        .merge(q3, on=['n', 'metric'])
        .rename(columns={'coverage': 'coverage_q3'})
    )

    for metric in df_plot['metric'].unique():
        df_metric = df_plot[df_plot['metric'] == metric]
        ax.plot(df_metric['n'], df_metric['coverage_median'], marker='o',
                color=color_dict[metric], linewidth=2, label=metric_labels[metric])
        ax.plot(df_metric['n'], df_metric['coverage_q1'], linestyle="--",
                color=color_dict[metric], linewidth=1)
        ax.plot(df_metric['n'], df_metric['coverage_q3'], linestyle="--",
                color=color_dict[metric], linewidth=1)
        ax.fill_between(df_metric['n'], df_metric['coverage_q1'], df_metric['coverage_q3'],
                        alpha=0.6, color=color_dict[metric])

    ax.set_title(f"Coverage — {stat_labels[stat]}", fontsize=22, weight="bold")
    ax.set_xlabel("Sample size", fontsize=16)
    ax.set_ylabel("Coverage", fontsize=16)
    ax.grid(True, axis="y")
    ax.set_ylim(None, 1.01)
    # --- Legends ---
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, title="Metrics", title_fontsize=15)

    # ============================================================
    # WIDTH (right column)
    # ============================================================
    ax = axs[row, 1]

    # --- CLASSIFICATION ---
    medians = df_macro_perc_width.groupby(['n', 'stat'])['width'].median().reset_index()
    q1 = df_macro_perc_width.groupby(['n', 'stat'])['width'].quantile(0.25).reset_index()
    q3 = df_macro_perc_width.groupby(['n', 'stat'])['width'].quantile(0.75).reset_index()
    df_plot = (
        medians.merge(q1, on=['n', 'stat'], suffixes=('_median', '_q1'))
        .merge(q3, on=['n', 'stat'])
        .rename(columns={'width': 'width_q3'})
    )

    for stat_classif in df_plot['stat'].unique():
        df_stat = df_plot[df_plot['stat'] == stat_classif]
        ax.plot(df_stat['n'], df_stat['width_median'], marker='o',
                color=color_dict_classif[stat_classif], linewidth=2,
                label=metric_labels[stat_classif])
        ax.plot(df_stat['n'], df_stat['width_q1'], linestyle="--",
                color=color_dict_classif[stat_classif], linewidth=1)
        ax.plot(df_stat['n'], df_stat['width_q3'], linestyle="--",
                color=color_dict_classif[stat_classif], linewidth=1)
        ax.fill_between(df_stat['n'], df_stat['width_q1'], df_stat['width_q3'],
                        alpha=0.2, color=color_dict_classif[stat_classif])

    # --- SEGMENTATION ---
    df_segm_mean = df_segm_perc_width[df_segm_perc_width['stat'] == stat]
    medians = df_segm_mean.groupby(['n', 'metric'])['width'].median().reset_index()
    q1 = df_segm_mean.groupby(['n', 'metric'])['width'].quantile(0.25).reset_index()
    q3 = df_segm_mean.groupby(['n', 'metric'])['width'].quantile(0.75).reset_index()
    df_plot = (
        medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
        .merge(q3, on=['n', 'metric'])
        .rename(columns={'width': 'width_q3'})
    )

    for metric in df_plot['metric'].unique():
        df_metric = df_plot[df_plot['metric'] == metric]
        ax.plot(df_metric['n'], df_metric['width_median'], marker='o',
                color=color_dict[metric], linewidth=2, label=metric_labels[metric])
        ax.plot(df_metric['n'], df_metric['width_q1'], linestyle="--",
                color=color_dict[metric], linewidth=1)
        ax.plot(df_metric['n'], df_metric['width_q3'], linestyle="--",
                color=color_dict[metric], linewidth=1)
        ax.fill_between(df_metric['n'], df_metric['width_q1'], df_metric['width_q3'],
                        alpha=0.7, color=color_dict[metric])

    ax.set_title(f"Width — {stat_labels[stat]}", fontsize=22, weight="bold")
    ax.set_xlabel("Sample size", fontsize=16)
    ax.set_ylabel("Width", fontsize=16)
    ax.grid(True, axis="y")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, title="Metrics", title_fontsize=15)

# Global layout
plt.suptitle("CI Comparison: Classification Macro vs Segmentation", fontsize=30, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../../../../journal paper plots/classif_vs_segm_all.pdf")


plt.show()
