"""
Build a task list for SLURM array jobs.
Uses OmegaConf.load() directly — no Hydra composition needed.

Usage:
    python src/utils/build_task_list.py \
        --ablation cfg/classif/ablations/epanechnikov_adaptive.yaml \
        --config-name classif/config_classif \
        --instance-dir instances_list \
        --output task_lists/epanechnikov_adaptive.txt
"""

import argparse, shlex, sys
from pathlib import Path
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True,
                        help="Path to ablation YAML file")
    parser.add_argument("--config-name", required=True,
                        help="Hydra config name for run.py overrides "
                             "(e.g. classif/config_classif)")
    parser.add_argument("--instance-dir", default="instances_list")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ablation_path = Path(args.ablation)
    if not ablation_path.is_file():
        print(f"ERROR: {ablation_path} not found", file=sys.stderr)
        sys.exit(1)

    # Ablation name for Hydra override (e.g. "epanechnikov_adaptive")
    ablation_name = ablation_path.stem

    # ── Load YAML directly — no Hydra ──────────────────────
    cfg = OmegaConf.load(ablation_path)
    sweep = OmegaConf.to_container(
        OmegaConf.select(cfg, "sweep"), resolve=True
    )

    # ── Detect pair type ────────────────────────────────────
    if "metric_average_pairs" in sweep:
        pairs = sweep["metric_average_pairs"]
        extra_key = "average"
    elif "metric_summary_pairs" in sweep:
        pairs = sweep["metric_summary_pairs"]
        extra_key = "summary_stat"
    else:
        print("ERROR: No metric_average_pairs or metric_summary_pairs "
              "found in sweep", file=sys.stderr)
        sys.exit(1)

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
                    f"ablations={ablation_name}"
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