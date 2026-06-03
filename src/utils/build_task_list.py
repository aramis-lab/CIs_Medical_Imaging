"""
Build a task list for SLURM array jobs.

Loads the sweep file referenced in run_plan.sweep_file,
crosses sweep pairs with instance lists.

Usage:
    python src/utils/build_task_list.py \
        --config-name classif/config_classif \
        --experiment epanechnikov_adaptive \
        --instance-dir instances_list \
        --output task_lists/epanechnikov_adaptive.txt
"""

import argparse, shlex, sys, os
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def find_config_dir():
    """Resolve src/cfg from script location."""
    return str(Path(__file__).resolve().parent.parent / "cfg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--instance-dir", default="instances_list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_dir = str(Path(args.config_dir or find_config_dir()).resolve())

    if not os.path.isdir(config_dir):
        print(f"ERROR: config dir not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Compose config with ablation ────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=args.config_name,
            overrides=[f"ablations={args.experiment}"],
        )

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # ── Load sweep file referenced in config ────────────────
    sweep_file = cfg.get("run_plan", {}).get("sweep_file")
    if sweep_file is None:
        print("ERROR: run_plan.sweep_file not found in config",
              file=sys.stderr)
        sys.exit(1)

    sweep_path = Path(config_dir) / sweep_file
    if not sweep_path.is_file():
        print(f"ERROR: Sweep file not found: {sweep_path}",
              file=sys.stderr)
        sys.exit(1)

    sweep = OmegaConf.to_container(
        OmegaConf.load(sweep_path), resolve=True
    )

    if args.debug:
        print(f"Config dir:  {config_dir}", file=sys.stderr)
        print(f"Sweep file:  {sweep_path}", file=sys.stderr)
        print(f"Sweep keys:  {list(sweep.keys())}", file=sys.stderr)

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
                    f"ablations={args.experiment}"
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