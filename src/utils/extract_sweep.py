"""Extract a list or scalar from a YAML config at a given dot-path.

A lightweight CLI helper used in shell scripts and CI pipelines to pull
values out of Hydra / OmegaConf configuration files without starting a
full Hydra run.  Depending on the type of the resolved value the output
differs:

* **Scalar** — printed as a single line.
* **List of scalars** — one element per line.
* **List of dicts** — one line per dict, with key–value pairs joined by
  semicolons (``key1=val1;key2=val2``), making it easy to parse in Bash
  with ``IFS=';'``.

Usage
-----
::

    python src/utils/extract_sweep.py run_plan.experiments \\
        --config src/cfg/classif/config_classif.yaml

    python src/utils/extract_sweep.py run_plan.sweep_file \\
        --config src/cfg/classif/config_classif.yaml
"""

import argparse, sys
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig


def main():
    """Parse CLI arguments, resolve the dot-path key, and print the result.

    The function performs three steps:

    1. **Load the YAML file** into an OmegaConf config object.
    2. **Select the value** at the user-supplied dot-path
       (e.g. ``run_plan.experiments``), resolving any interpolations.
    3. **Print the value** to *stdout* in a format suited to its type
       (see module docstring for formatting rules).

    Raises
    ------
    SystemExit
        If the config file does not exist or the requested key is not
        found in the configuration.
    """
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

    # Convert OmegaConf objects to plain Python, leave scalars as-is
    if isinstance(values, (DictConfig, ListConfig)):
        values = OmegaConf.to_container(values, resolve=True)

    if isinstance(values, list):
        for v in values:
            if isinstance(v, dict):
                print(";".join(f"{k}={val}" for k, val in v.items()))
            else:
                print(v)
    else:
        # Scalar (string, int, bool, etc.)
        print(values)


if __name__ == "__main__":
    main()