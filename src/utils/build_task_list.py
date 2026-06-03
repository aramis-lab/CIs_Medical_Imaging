"""
Build a task list for SLURM array jobs.

Usage:
    python src/utils/build_task_list.py \
        --ablation-name epanechnikov_adaptive \
        --ablation-group classif/ablations \
        --sweep src/cfg/sweep/classif_all_pairs.yaml \
        --instance-dir instances_list \
        --output task_lists/epanechnikov_adaptive.txt
"""

import argparse, shlex, sys
from pathlib import Path
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-name", required=True,
                        help="e.g. epanechnikov_adaptive")
    parser.add_argument("--ablation-group", required=True,
                        help="Hydra config group path, e.g. classif/ablations")
    parser.add_argument("--sweep", required=True,
                        help="Path to sweep YAML file")
    parser.add_argument("--instance-dir", default="instances_list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.is_file():
        print(f"ERROR: {sweep_path} not found", file=sys.stderr)
        sys.exit(1)

    sweep = OmegaConf.to_container(
        OmegaConf.load(sweep_path), resolve=True
    )

    if args.debug:
        print(f"Ablation:    {args.ablation_name}", file=sys.stderr)
        print(f"Group:       {args.ablation_group}", file=sys.stderr)
        print(f"Sweep file:  {sweep_path}", file=sys.stderr)

    # ── Detect pair type ────────────────────────────────────
    if "metric_average_pairs" in sweep:
        pairs = sweep["metric_average_pairs"]
        extra_key = "average"
    elif "metric_summary_pairs" in sweep:
        pairs = sweep["metric_summary_pairs"]
        extra_key = "summary_stat"
    else:
        print("ERROR: No metric_average_pairs or metric_summary_pairs "
              "in sweep file", file=sys.stderr)
        sys.exit(1)

    # ── Hydra override for the ablation ─────────────────────
    # e.g. 'classif/ablations@ablations=epanechnikov_adaptive'
    group = args.ablation_group
    group_name = group.split("/")[-1]  # "ablations"
    ablation_override = f"'{group}@{group_name}={args.ablation_name}'"

    # ── Cross pairs × instance lists ────────────────────────
    instance_dir = Path(args.instance_dir)
    lines = []

    for pair in pairs:
        metric = pair["metric"]
        extra_val = pair[extra_key]
        instance_file = instance_dir / f"{metric}.txt"

        if not instance_file.exists():
            print(f"WARNING: {instance_file} not found, skipping",
                  file=sys.stderr)
            continue

        with open(instance_file) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                task, algo = raw.split(maxsplit=1)
                overrides = (
                    f"{ablation_override}"
                    f" metric={metric}"
                    f" {extra_key}={extra_val}"
                    f" +task={shlex.quote(task)}"
                    f" +algo={shlex.quote(algo)}"
                )
                lines.append(overrides)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} tasks → {output_path}")


if __name__ == "__main__":
    main()