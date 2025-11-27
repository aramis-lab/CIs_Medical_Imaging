# Choose layout: each metric gets 2 columns (coverage + width)
means_macro_df_n=df_macro[(df_macro['method']=='percentile') & (df_macro['n']<=250)].sort_values(by=['stat', 'n'])
means_micro_df_n=df_micro[(df_micro['method']=='percentile') & (df_micro['n']<=250)].sort_values(by=['stat', 'n'])

means_macro_df_n_width =df_macro_width[(df_macro_width['method']=='percentile') & (df_macro_width['n']<=250)].sort_values(by=['stat', 'n'])
means_micro_df_n_width =df_micro_width[(df_micro_width['method']=='percentile') & (df_micro_width['n']<=250)].sort_values(by=['stat', 'n'])

metrics_classif=np.append(means_macro_df_n['stat'].unique(), 'accuracy')
palette_classif = sns.color_palette("colorblind", len(metrics_classif))
color_dict_classif = dict(zip(metrics_classif, palette))
num_metrics = len([m for m in metrics_classif if m not in ['accuracy', 'balanced_accuracy']]) + 1
fig, axes = plt.subplots(
    num_metrics, 2, 
    figsize=(18, 6 * num_metrics), 
    sharex=True
)

# If only one row, force axes to be 2D
if num_metrics == 1:
    axes = np.array([axes])

row = 0

for metric in metrics_classif:
    if metric in ['accuracy', 'balanced_accuracy']:
        continue

    ax_cov = axes[row, 0]
    ax_width = axes[row, 1]

    # Coverage ----------------------------------------------------------
    macro_data = means_macro_df_n[means_macro_df_n['stat'] == metric]
    micro_data = means_micro_df_n[means_micro_df_n['stat'] == metric]

    for data, label in [(macro_data, "Macro"), (micro_data, "Micro")]:
        if not data.empty:
            med = data.groupby("n")["value"].median()
            q1 = data.groupby("n")["value"].quantile(0.25)
            q3 = data.groupby("n")["value"].quantile(0.75)
            n_vals = med.index.values

            ax_cov.plot(n_vals, med.values, marker="o", label=label)
            ax_cov.fill_between(n_vals, q1.values, q3.values, alpha=0.2)

    ax_cov.set_title(f"Coverage – {metric_labels[metric]}", weight="bold")
    ax_cov.set_xlabel("Sample size")
    ax_cov.set_ylabel("Coverage")
    ax_cov.grid(True, axis='y')
    ax_cov.set_ylim(None, 1.01)
    ax_cov.legend()

    # Width subplot (right)
    # Macro
    macro_width = means_macro_df_n_width[means_macro_df_n_width['stat'] == metric]
    if not macro_width.empty:
        medians = macro_width.groupby('n')['width'].median().values
        q1 = macro_width.groupby('n')['width'].quantile(0.25).values
        q3 = macro_width.groupby('n')['width'].quantile(0.75).values
        n_vals = macro_width['n'].unique()
        ax_width.plot(n_vals, medians, marker='o', label='Macro', linewidth=2, markersize=8)
        ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
    # Micro
    micro_width = means_micro_df_n_width[means_micro_df_n_width['stat'] == metric]
    if not micro_width.empty:
        medians = micro_width.groupby('n')['width'].median().values
        q1 = micro_width.groupby('n')['width'].quantile(0.25).values
        q3 = micro_width.groupby('n')['width'].quantile(0.75).values
        n_vals = micro_width['n'].unique()
        ax_width.plot(n_vals, medians, marker='o', label='Micro', linewidth=2, markersize=8)
        ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
    ax_width.set_title(f'Width for {metric_labels[metric]}', weight='bold')
    ax_width.set_xlabel('Sample size', weight='bold', fontsize=14)
    ax_width.set_ylabel('Width', weight='bold', fontsize=14)
    ax_width.tick_params(axis='y', labelsize=12)
    ax_width.tick_params(axis='x', labelsize=12)
    ax_width.set_ylim(-0.01, None)
    ax_width.grid(True, axis='y')
    ax_width.legend()

    row += 1


# ----------- Balanced Accuracy vs Accuracy (final row) ------------------

ax_cov = axes[row, 0]
ax_width = axes[row, 1]

# Coverage
macro_data = means_macro_df_n[means_macro_df_n['stat'] == "balanced_accuracy"]
micro_data = means_micro_df_n[means_micro_df_n['stat'] == "accuracy"]

# Width subplot (right)
# Macro
macro_width = means_macro_df_n_width[means_macro_df_n_width['stat'] == "balanced_accuracy"]
if not macro_width.empty:
    medians = macro_width.groupby('n')['width'].median().values
    q1 = macro_width.groupby('n')['width'].quantile(0.25).values
    q3 = macro_width.groupby('n')['width'].quantile(0.75).values
    n_vals = macro_width['n'].unique()
    ax_width.plot(n_vals, medians, marker='o', label='Balanced Accuracy', linewidth=2, markersize=8)
    ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
# Micro
micro_width = means_micro_df_n_width[means_micro_df_n_width['stat'] == "accuracy"]
if not micro_width.empty:
    medians = micro_width.groupby('n')['width'].median().values
    q1 = micro_width.groupby('n')['width'].quantile(0.25).values
    q3 = micro_width.groupby('n')['width'].quantile(0.75).values
    n_vals = micro_width['n'].unique()
    ax_width.plot(n_vals, medians, marker='o', label='Accuracy', linewidth=2, markersize=8)
    ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
ax_width.set_title(f'Width for {metric_labels["balanced_accuracy"]} vs Accuracy', weight='bold')
ax_width.set_xlabel('Sample size', weight='bold', fontsize=14)
ax_width.set_ylabel('Width', weight='bold', fontsize=14)
ax_width.tick_params(axis='y', labelsize=12)
ax_width.tick_params(axis='x', labelsize=12)
ax_width.set_ylim(-0.01, None)
ax_width.grid(True, axis='y')
ax_width.legend()

plt.tight_layout()
plt.savefig("../../../../journal paper plots/classification/aggregation method/micro_vs_macro_all.pdf")
plt.show()
