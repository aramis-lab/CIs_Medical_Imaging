from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

################################################################################
# PARAMETERS
################################################################################

METHODS_SEGMENTATION = ['bca', 'basic', 'percentile', 'param_t']
METHODS_CLASSIF = ['bca', 'basic', 'percentile', 'wilson']

METRICS_SEGMENTATION = [
    'dsc',
    'nsd',
    'iou',
    'boundary_iou',
    'assd',
    'masd',
    'hd',
    'hd_perc'
]

METRICS_CLASSIF_MICRO = [
    'accuracy',
    'ap',
    'auc',
    'f1_score'
]

METRICS_CLASSIF_MACRO = [
    'balanced_accuracy',
    'ap',
    'auc',
    'f1_score'
]

SUMMARY_STATS = ["mean", "median", 'trimmed_mean']

INPUT_DIR = Path("../results_metrics_segm")
OUTPUT_MD = Path("coverage_tree.md")

################################################################################
# UTILITIES
################################################################################

def lower_whisker(values):
    """
    Tukey lower whisker.
    """
    values = np.asarray(values)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    limit = q1 - 1.5 * iqr

    valid = values[values >= limit]

    return np.min(valid)


def classify(typical, worst):
    """
    Return recommendation tag and coverage values.
    """

    # good methods
    if typical > 0.92 and worst > 0.90:
        return "✅", None

    # acceptable typical but bad worst case
    if typical > 0.92 and worst < 0.90:
        return "⚠️", f"{typical*100:.0f}/{worst*100:.0f}"

    # not recommended
    return None, None



################################################################################
# READ ALL FILES
################################################################################

records = []

for metric in (
    METRICS_SEGMENTATION
    
):
    for stat in SUMMARY_STATS:

        file = INPUT_DIR / f"aggregated_results_{metric}_{stat}.csv"

        if not file.exists():
            continue

        df = pd.read_csv(file)

        df["metric"] = metric
        df["summary_stat"] = stat

        records.append(df)

if len(records) == 0:
    raise RuntimeError("No CSV files found.")

df = pd.concat(records, ignore_index=True)

################################################################################
# COMPUTE METHOD STATUS
################################################################################
status = defaultdict(dict)

for (stat, metric, n), g in df.groupby(
    ['summary_stat', 'metric', 'n']
):

    methods = METHODS_SEGMENTATION[:-1] if stat != 'mean' else METHODS_SEGMENTATION

    recommended = {}
    warnings = {}

    for method in methods:

        typical = np.median(
            g[f'contains_true_stat_{method}']
        )

        worst = lower_whisker(
            g[f'contains_true_stat_{method}']
        )

        # Best case
        if typical > 0.92 and worst > 0.90:

            recommended[method] = (
                "✅",
                None
            )

        # Warning case
        elif typical > 0.92 and worst < 0.90:

            warnings[method] = (
                "⚠️",
                f"{typical*100:.0f}/{worst*100:.0f}"
            )


    # Priority:
    # 1) keep only green methods
    if len(recommended) > 0:

        status[(stat, metric, n)] = recommended


    # 2) otherwise keep warnings
    elif len(warnings) > 0:

        status[(stat, metric, n)] = warnings


    # 3) otherwise red cross
    else:

        status[(stat, metric, n)] = {
            "__none__": ("❌", None)
        }
################################################################################
# BUILD LEAF SIGNATURES
################################################################################

leaf_signature = {}

for key, methods in status.items():

    ordered = tuple(sorted(methods.items()))

    leaf_signature[key] = ordered

################################################################################
# GROUP IDENTICAL LEAVES
################################################################################

groups = defaultdict(list)

for key, sig in leaf_signature.items():
    groups[sig].append(key)

################################################################################
# FORMAT METHODS
################################################################################

def methods_markdown(sig):

    lines = []

    for method, tag in sig:

      
        lines.append(f"- {tag} {method}")

    return lines


def node_signature(node):
    """
    Return a hashable representation of a tree node.
    Used to identify identical subtrees.
    """
    if isinstance(node, dict):
        return tuple(
            sorted(
                (k, node_signature(v))
                for k, v in node.items()
            )
        )

    return node


def merge_identical_nodes(node):

    if not isinstance(node, dict):
        return node

    # First compress children recursively
    compressed = defaultdict(list)

    for name, child in node.items():

        child = merge_identical_nodes(child)

        compressed[node_signature(child)].append(
            (name, child)
        )


    merged = {}

    for _, items in compressed.items():

        names = []

        # all nodes have identical children
        for name, child in items:
            names.append(str(name))

        
            # sort sample sizes numerically, other names alphabetically
        if all(str(x).isdigit() for x in names):
            merged_name = ", ".join(
                map(str, sorted(map(int, names)))
            )
        else:
            merged_name = ", ".join(
                sorted(names)
            )  

        merged[merged_name] = items[0][1]

    return merged
def write_tree(node, level=1):

    lines = []

    if not isinstance(node, dict):
        return lines
    def sort_key(item):
        name = str(item[0])

        if name.isdigit():
            return (0, int(name))

        return (1, name)


    for name, child in sorted(node.items(), key=sort_key):

        # format sample size nodes
        if str(name).isdigit():
            display_name = f"{name}"
        else:
            display_name = str(name)

        lines.append(
            "  " * level + f"- {display_name}"
        )

        # method leaves
        if isinstance(child, dict) and all(
            isinstance(v, tuple) for v in child.values()
        ):

            for method, (tag, value) in sorted(child.items()):

                # red cross case: no methods
                if method == "__none__":
                    lines.append(
                        "  " * (level + 1)
                        + "- ❌"
                    )

                # green case
                elif value is None:
                    lines.append(
                        "  " * (level + 1)
                        + f"- {tag} {method}"
                    )

                # warning case with typical/worst
                else:
                    lines.append(
                        "  " * (level + 1)
                        + f"- {tag} {method} ({value})"
                    )

        else:
            lines.extend(
                write_tree(child, level + 1)
            )

    return lines


################################################################################
# WRITE TREE
################################################################################


tree = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))



for (stat, metric, n), methods in status.items():
    tree[stat][metric][n] = methods

tree = merge_identical_nodes(tree)

if OUTPUT_MD.exists():
    OUTPUT_MD.unlink()
out = []

out.append("- Segmentation")


out.extend(write_tree(tree))

OUTPUT_MD.write_text(
    "\n".join(out),
    encoding="utf8"
)


# for stat in sorted(tree.keys()):

#     out.append(f"   - {stat}")

#     for metric in sorted(tree[stat].keys()):

#         out.append(f"       - {metric}")

#         for n in sorted(tree[stat][metric]):

#             out.append(f"           - n={n}")

#             methods = tree[stat][metric][n]

#             for method, tag in sorted(methods.items()):

#                 out.append(
#                     f"              - {tag} {method}"
#                 )

#     out.append("")


# OUTPUT_MD.write_text("\n".join(out), encoding="utf8")

print(f"Markdown written to {OUTPUT_MD}")