"""
Extract a list from a YAML config at a given dot-path.

Usage:
    python src/utils/extract_sweep.py run_plan.experiments \
        --config src/cfg/classif/config_classif.yaml

    python src/utils/extract_sweep.py sweep.metric_average_pairs \
        --config src/cfg/classif/ablations/epanechnikov_adaptive.yaml
"""

import argparse, sys
from pathlib import Path
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="Dot-path, e.g. run_plan.experiments")
    parser.add_argument("--config", required=True,
                        help="Path to YAML file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: File not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    if args.debug:
        print(f"File: {config_path}", file=sys.stderr)
        print(f"Top-level keys: {list(cfg.keys())}", file=sys.stderr)
        print(OmegaConf.to_yaml(cfg), file=sys.stderr)

    values = OmegaConf.select(cfg, args.key)

    if values is None:
        print(f"ERROR: Key '{args.key}' not found.", file=sys.stderr)
        print(f"Available keys: {list(cfg.keys())}", file=sys.stderr)
        sys.exit(1)
    
    values = OmegaConf.to_container(values, resolve=True)

    if isinstance(values, list):
        for v in values:
            if isinstance(v, dict):
                print(";".join(f"{k}={val}" for k, val in v.items()))
            else:
                print(v)
    else:
        print(values)


if __name__ == "__main__":
    main()