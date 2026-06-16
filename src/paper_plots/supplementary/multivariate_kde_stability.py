import numpy as np
from scipy.stats import ks_2samp, rankdata
import matplotlib.pyplot as plt
import argparse
import os

from ...kde import sample_weighted_kde_multivariate
from ...kernels import get_kernel
from ...intervals_and_metrics import softmax, get_metric, label_binarize_vectorized
from ...utils import extract_df

# ─────────────────────────────────────────────────────────────
#  Core diagnostics
# ─────────────────────────────────────────────────────────────

def concordance_probability(pos_scores, neg_scores, n_pairs=500_000):
    if len(pos_scores) < 2 or len(neg_scores) < 2:
        return np.nan, np.nan
    idx_p = np.random.randint(0, len(pos_scores), n_pairs)
    idx_n = np.random.randint(0, len(neg_scores), n_pairs)
    comparisons = (pos_scores[idx_p] > neg_scores[idx_n]).astype(np.float64)
    comparisons += 0.5 * (pos_scores[idx_p] == neg_scores[idx_n]).astype(np.float64)
    return float(np.mean(comparisons)), float(np.std(comparisons) / np.sqrt(n_pairs))


def normalized_rank_positions_of_positives(pos_scores, neg_scores):
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    ranks = rankdata(-all_scores)  # descending
    n = len(all_scores)
    return (ranks[labels == 1] - 1) / max(n - 1, 1)


def precision_at_k_curve(pos_scores, neg_scores, max_k=500):
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    max_k = min(max_k, len(all_scores))
    sorted_labels = labels[np.argsort(-all_scores)][:max_k]
    k = np.arange(1, max_k + 1)
    return k, np.cumsum(sorted_labels) / k

def precision_at_fraction_curve(pos_scores, neg_scores, fractions):      
    """                                                                   
    Compute precision at specified fractions of the ranked list.          
    Both original and KDE curves can be evaluated on the same            
    fraction grid, making them visually comparable regardless of n.      
    """                                                                   
    all_scores = np.concatenate([pos_scores, neg_scores])                 
    labels = np.concatenate([np.ones(len(pos_scores)),                    
                             np.zeros(len(neg_scores))])                  
    n = len(all_scores)                                                   
    sorted_labels = labels[np.argsort(-all_scores)]                       
    cumsum = np.cumsum(sorted_labels)                                     
                                                                    
    k_values = np.unique(np.clip((fractions * n).astype(int), 1, n))      
    precisions = cumsum[k_values - 1] / k_values                         
    actual_fractions = k_values / n                                       
                                                                    
    return actual_fractions, precisions                                   


def compute_roc_points(pos_scores, neg_scores, n_thresholds=500):
    """Memory-safe ROC via searchsorted."""
    lo = min(pos_scores.min(), neg_scores.min()) - 1e-10
    hi = max(pos_scores.max(), neg_scores.max()) + 1e-10
    thresholds = np.linspace(hi, lo, n_thresholds)
    sorted_pos = np.sort(pos_scores)
    sorted_neg = np.sort(neg_scores)
    n_pos, n_neg = len(sorted_pos), len(sorted_neg)
    tpr = (n_pos - np.searchsorted(sorted_pos, thresholds, side='left')) / n_pos
    fpr = (n_neg - np.searchsorted(sorted_neg, thresholds, side='left')) / n_neg
    return fpr, tpr


def _split_pos_neg_class(scores, labels_bin, c):
    """Extract positive and negative scores for class c (OvR)."""
    mask_pos = labels_bin[:, c] == 1
    return scores[mask_pos, c], scores[~mask_pos, c]


def _split_pos_neg_micro(scores, labels_bin):
    """Flatten for micro averaging."""
    flat_s = scores.ravel()
    flat_l = labels_bin.ravel().astype(int)
    return flat_s[flat_l == 1], flat_s[flat_l == 0]


# ─────────────────────────────────────────────────────────────
#  Main analysis
# ─────────────────────────────────────────────────────────────

def analyze_score_ordering_preservation(
    scores, labels_bin, kde_scores, kde_labels_bin, n_classes,
    task="", algo=""
):
    results = dict(task=task, algo=algo, n_classes=n_classes,
                   n_orig=len(scores), n_kde=len(kde_scores),
                   per_class={}, micro={})

    # ── per-class ──
    for c in range(n_classes):
        orig_pos, orig_neg = _split_pos_neg_class(scores, labels_bin, c)
        kde_pos, kde_neg   = _split_pos_neg_class(kde_scores, kde_labels_bin, c)

        cr = {}
        cr['conc_orig'], cr['conc_orig_se'] = concordance_probability(orig_pos, orig_neg)
        cr['conc_kde'],  cr['conc_kde_se']  = concordance_probability(kde_pos, kde_neg)
        cr['conc_abs_diff'] = (abs(cr['conc_orig'] - cr['conc_kde'])
                               if not (np.isnan(cr['conc_orig']) or np.isnan(cr['conc_kde']))
                               else np.nan)

        cr['ks_pos'] = dict(zip(['stat', 'p'],
                    ks_2samp(orig_pos, kde_pos))) if len(orig_pos) > 1 and len(kde_pos) > 1 \
                    else dict(stat=np.nan, p=np.nan)

        cr['ks_neg'] = dict(zip(['stat', 'p'],
                    ks_2samp(orig_neg, kde_neg))) if len(orig_neg) > 1 and len(kde_neg) > 1 \
                    else dict(stat=np.nan, p=np.nan)

        if len(orig_pos) > 0 and len(orig_neg) > 0 and len(kde_pos) > 0 and len(kde_neg) > 0:
            rp_orig = normalized_rank_positions_of_positives(orig_pos, orig_neg)
            rp_kde  = normalized_rank_positions_of_positives(kde_pos, kde_neg)
            cr['rank_ks'] = dict(zip(['stat', 'p'], ks_2samp(rp_orig, rp_kde)))
            cr['rank_mean_orig'] = float(np.mean(rp_orig))
            cr['rank_mean_kde']  = float(np.mean(rp_kde))
        else:
            cr['rank_ks'] = dict(stat=np.nan, p=np.nan)
            cr['rank_mean_orig'] = cr['rank_mean_kde'] = np.nan

        cr['n_pos_orig'] = len(orig_pos)
        cr['n_pos_kde']  = len(kde_pos)
        results['per_class'][c] = cr

    # ── micro ──
    orig_pos_m, orig_neg_m = _split_pos_neg_micro(scores, labels_bin)
    kde_pos_m, kde_neg_m   = _split_pos_neg_micro(kde_scores, kde_labels_bin)

    mi = results['micro']
    mi['conc_orig'], mi['conc_orig_se'] = concordance_probability(orig_pos_m, orig_neg_m)
    mi['conc_kde'],  mi['conc_kde_se']  = concordance_probability(kde_pos_m, kde_neg_m)
    mi['conc_abs_diff'] = abs(mi['conc_orig'] - mi['conc_kde'])

    mi['ks_pos'] = dict(zip(['stat', 'p'], ks_2samp(orig_pos_m, kde_pos_m)))
    mi['ks_neg'] = dict(zip(['stat', 'p'], ks_2samp(orig_neg_m, kde_neg_m)))

    rp_orig = normalized_rank_positions_of_positives(orig_pos_m, orig_neg_m)
    rp_kde  = normalized_rank_positions_of_positives(kde_pos_m, kde_neg_m)
    mi['rank_ks'] = dict(zip(['stat', 'p'], ks_2samp(rp_orig, rp_kde)))

    return results


# ─────────────────────────────────────────────────────────────
#  Print & Plot (unchanged but using the safe helpers)
# ─────────────────────────────────────────────────────────────

def print_ordering_summary(res):
    print(f"\n{'='*90}")
    print(f"  SCORE ORDERING PRESERVATION — {res['task']} / {res['algo']}")
    print(f"  n_orig={res['n_orig']}  n_kde={res['n_kde']}  n_classes={res['n_classes']}")
    print(f"{'='*90}")

    mi = res['micro']
    print(f"\n  MICRO  P(s⁺>s⁻) — orig: {mi['conc_orig']:.6f} ± {mi['conc_orig_se']:.1e}"
          f"   kde: {mi['conc_kde']:.6f} ± {mi['conc_kde_se']:.1e}"
          f"   |Δ|={mi['conc_abs_diff']:.6f}")
    print(f"         KS(f⁺): stat={mi['ks_pos']['stat']:.4f}"
          f"   KS(f⁻): stat={mi['ks_neg']['stat']:.4f}"
          f"   KS(rank): stat={mi['rank_ks']['stat']:.4f}")
    print(f"  NOTE: With large n, focus on KS statistics (<0.05 excellent), not p-values.\n")

    header = (f"  {'Cls':>4} {'n⁺':>6} | {'Conc(orig)':>10} {'Conc(kde)':>10} {'|Δ|':>8}"
              f" | {'KS(f⁺)':>7} {'KS(f⁻)':>7} {'KS(rank)':>8}")
    print(header)
    print(f"  {'-'*80}")

    diffs = []
    for c in range(res['n_classes']):
        pc = res['per_class'][c]
        d = pc['conc_abs_diff']
        diffs.append(d)
        print(f"  {c:>4} {pc['n_pos_orig']:>6} |"
              f" {pc['conc_orig']:>10.6f} {pc['conc_kde']:>10.6f} {d:>8.6f}"
              f" | {pc['ks_pos']['stat']:>7.4f} {pc['ks_neg']['stat']:>7.4f}"
              f" {pc['rank_ks']['stat']:>8.4f}")

    print(f"\n  Mean |Δ concordance| across classes: {np.nanmean(diffs):.6f}")
    print(f"  Max  |Δ concordance| across classes: {np.nanmax(diffs):.6f}")
    print(f"{'='*90}\n")


def plot_ordering_diagnostics(
    scores, labels_bin, kde_scores, kde_labels_bin,
    n_classes, task="", algo="",
    max_classes_to_plot=4, save_path=None
):
    pos_counts = labels_bin.sum(axis=0)
    plot_classes = np.argsort(-pos_counts)[:max_classes_to_plot]

    n_rows = 1 + len(plot_classes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Score Ordering Diagnostics — {task} / {algo}",
                 fontsize=14, fontweight='bold', y=1.01)

    # Common fraction grid for Precision@k
    fractions = np.linspace(0.001, 0.7, 500)

    def _row(ax_row, o_pos, o_neg, k_pos, k_neg, label):
        # ROC
        ax = ax_row[0]
        fpr_o, tpr_o = compute_roc_points(o_pos, o_neg)
        fpr_k, tpr_k = compute_roc_points(k_pos, k_neg)
        ax.plot(fpr_o, tpr_o, 'b-',  lw=2, label='Original')
        ax.plot(fpr_k, tpr_k, 'r--', lw=2, label='KDE')
        ax.plot([0, 1], [0, 1], 'k:', alpha=.3)
        ax.set(xlabel='FPR', ylabel='TPR', title=f'{label} — ROC')
        ax.legend(fontsize=8)

        # Rank distribution
        ax = ax_row[1]
        rp_o = normalized_rank_positions_of_positives(o_pos, o_neg)
        rp_k = normalized_rank_positions_of_positives(k_pos, k_neg)
        bins = np.linspace(0, 1, 51)
        ax.hist(rp_o, bins=bins, density=True, alpha=.5, color='blue',  label='Original')
        ax.hist(rp_k, bins=bins, density=True, alpha=.5, color='red',   label='KDE')
        ax.set(xlabel='Normalized rank', ylabel='Density',
               title=f'{label} — Positive rank dist.')
        ax.legend(fontsize=8)

        # Precision@k
        ax = ax_row[2]
        f_o, p_o = precision_at_fraction_curve(o_pos, o_neg, fractions)
        f_k, p_k = precision_at_fraction_curve(k_pos, k_neg, fractions)
        ax.plot(f_o, p_o, 'b-',  lw=2, label='Original')
        ax.plot(f_k, p_k, 'r--', lw=2, label='KDE')
        ax.set(xlabel='Fraction examined', ylabel='Precision@k',
               title=f'{label} — Precision@k')
        ax.legend(fontsize=8)

    # micro row
    o_pos_m, o_neg_m = _split_pos_neg_micro(scores, labels_bin)
    k_pos_m, k_neg_m = _split_pos_neg_micro(kde_scores, kde_labels_bin)
    _row(axes[0], o_pos_m, o_neg_m, k_pos_m, k_neg_m, "Micro")

    # per-class rows
    for i, c in enumerate(plot_classes):
        op, on = _split_pos_neg_class(scores, labels_bin, c)
        kp, kn = _split_pos_neg_class(kde_scores, kde_labels_bin, c)
        _row(axes[i + 1], op, on, kp, kn, f"Class {c}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def empirical_stability_analysis(df_path, output_folder, task, algo):

    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Data matrix not found at {df_path}. Please ensure the file exists.")
    results = {}
    for metric in ["auc", "ap"]:

        df = extract_df(df_path, metric, task)

        metric_func = get_metric(metric)

        # Convert string representations of sets to 2D numpy array
        logits_str = df[df["alg_name"].astype(str) == algo]["logits"]
        values = [list(eval(v, {"nan": np.nan})) for v in logits_str]
        if len(values) == 0:
            print(f"Not enough values for {task} {algo} ({len(values)}), skipping KDE")
            return
        lengths = np.array([len(v) for v in values])
        good_length = round(np.mean(lengths))
        indices = np.where(lengths==good_length)
        values = np.array([v for v in values if len(v)==good_length])
        labels = df[df["alg_name"].astype(str) == algo]["target"].to_numpy()[indices]

        kernel = get_kernel("epanechnikov")

        # Define the grid for KDE
        alphas = np.ones(len(values))

        # Iterative weighted KDE estimation
        initial_estimates = kernel(values, values, alphas)
        initial_estimates = np.mean(initial_estimates, axis=1)
        log_g = np.mean(np.log(initial_estimates))
        g = np.exp(log_g)
        alphas = (initial_estimates / g) ** (-1/2)
        kde_values, kde_labels = sample_weighted_kde_multivariate(values, labels, "epanechnikov", 100000, alphas)
        kde_scores = softmax(kde_values)
    
        n_classes = kde_scores.shape[-1]

        kde_labels_bin = label_binarize_vectorized(kde_labels, n_classes)
        labels_bin = label_binarize_vectorized(labels, n_classes)

        scores = softmax(values)

        orig_micro = metric_func(scores, labels_bin, average="micro")
        orig_macro = metric_func(scores, labels_bin, average="macro")

        kde_micro = metric_func(kde_scores, kde_labels_bin, average="micro")
        kde_macro = metric_func(kde_scores, kde_labels_bin, average="macro")

        results[metric] = {
            "orig_micro": orig_micro,
            "orig_macro": orig_macro,
            "kde_micro": kde_micro,
            "kde_macro": kde_macro
        }

    ordering_results = analyze_score_ordering_preservation(
        scores, labels_bin, kde_scores, kde_labels_bin,
        n_classes, task=task, algo=algo
    )

    # Print summary table
    print_ordering_summary(ordering_results)

    # Contextualize with actual metric values
    print(f"  Metric comparison:")
    for metric, values in results.items():
        print(f"    {metric.upper()} micro — orig: {values['orig_micro']:.6f}  kde: {values['kde_micro']:.6f}"
              f"  |Δ|={abs(values['orig_micro'] - values['kde_micro']):.6f}")
        print(f"    {metric.upper()} macro — orig: {values['orig_macro']:.6f}  kde: {values['kde_macro']:.6f}"
              f"  |Δ|={abs(values['orig_macro'] - values['kde_macro']):.6f}")
        print(f"    AP  macro — orig: {values['orig_macro']:.6f}  kde: {values['kde_macro']:.6f}"
              f"  |Δ|={abs(values['orig_macro'] - values['kde_macro']):.6f}")

    # Diagnostic plots (optional — comment out for batch runs)
    plot_ordering_diagnostics(
        scores, labels_bin, kde_scores, kde_labels_bin,
        n_classes, task=task, algo=algo,
        save_path=os.path.join(output_folder, f"ordering_diagnostics_{task}_{algo}.pdf")
    )

    return ordering_results

if __name__ == "__main__":
    # Load the data matrix for classification
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    df_name = "data_matrix_classification.csv"

    df_path = os.path.join(BASE_DIR, df_name)
    
    parser = argparse.ArgumentParser(description="Run empirical stability analysis on classification data.")
    parser.add_argument("--task", type=str, required=True, help="Task name for analysis.")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name for analysis.")
    parser.add_argument("--output_folder", type=str, default=BASE_DIR, help="Directory to save output plots and results.")
    args = parser.parse_args()

    empirical_stability_analysis(df_path, args.output_folder, args.task, args.algo)