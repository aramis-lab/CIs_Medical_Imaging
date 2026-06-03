"""
Extract a list from Hydra config at a given dot-path.

Usage:
    python src/utils/extract_sweep.py run_plan.experiments \
        --config-name=config_classif
    python src/utils/extract_sweep.py run_plan.all_metrics \
        --config-name=config_classif
"""

import argparse
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="Dot-path, e.g. run_plan.experiments")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides")
    args = parser.parse_args()

    config_dir = args.config_dir or str(
        Path(__file__).resolve().parents[1] / "cfg"
    )

    cfg = OmegaConf.load(config_dir + "/" + args.config_name + ".yaml")
    values = OmegaConf.select(cfg, args.key)

    if values is None:
        raise KeyError(f"Key '{args.key}' not found in config")

    if isinstance(values, (list, tuple)):
        for v in values:
            print(v)
    else:
        print(values)


if __name__ == "__main__":
    main()