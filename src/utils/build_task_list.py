"""
Build a task list for SLURM array jobs.

For each sweep pair x each (task, algo) from instance lists,
produce one line of Hydra overrides.

Auto-detects classification (metric_average_pairs) vs 
segmentation (metric_summary_pairs).

Usage:
    python src/utils/build_task_list.py \
        --config-name config_classif \
        --experiment epanechnikov_adaptive \
        --instance-dir instances_list \
        --output task_lists/epanechnikov_adaptive.txt
"""

import argparse
import shlex
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--instance-dir", default="instances_list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    config_dir = args.config_dir or str(
        Path(__file__).resolve().parents[1] / "cfg"
    )

    # ── Load config with ablation ───────────────────────────
    if args.config_name.startswith("classif/"):
        ablation_override = f"+classif/ablations={args.experiment}"
    elif args.config_name.startswith("segm/"):
        ablation_override = f"+segm/ablations={args.experiment}"
    else:
        ablation_override = f"+ablations={args.experiment}"

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=args.config_name,
            overrides=[ablation_override],
        )

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
        raise ValueError(
            "Expected 'metric_average_pairs' or 'metric_summary_pairs' "
            "in sweep config"
        )

    # ── Cross pairs × instance lists ────────────────────────
    instance_dir = Path(args.instance_dir)
    lines = []

    for pair in pairs:
        metric = pair["metric"]
        extra_val = pair[extra_key]
        instance_file = instance_dir / f"{metric}.txt"

        if not instance_file.exists():
            print(f"WARNING: {instance_file} not found, skipping {pair}")
            continue

        with open(instance_file) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                task, algo = raw.split(maxsplit=1)

                # shlex.quote handles spaces / special chars safely
                overrides = (
                    f"{ablation_override}"
                    f" metric={metric}"
                    f" {extra_key}={extra_val}"
                    f" +task={shlex.quote(task)}"
                    f" +algo={shlex.quote(algo)}"
                )
                lines.append(overrides)

    # ── Write task list ─────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} tasks → {output_path}")


if __name__ == "__main__":
    main()