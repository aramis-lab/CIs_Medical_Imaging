# CIs_Medical_Imaging
This is the repository for the Confidence Interval in Medical Imaging project.

## Getting Started

There are two ways to use this repo, depending on what you need:

- **Single experiment**: you have one metric, one task, and want the coverage of the confidence-interval methods, without any SLURM setup. Use `run_single.py`.
- **Full pipeline**: you want to reproduce the paper's experiments across every metric/average/summary-stat pair, ablation, task, and algorithm, submitted as SLURM array jobs. Use `run_all.sh`.

Both pipelines share the same underlying code (`src/run.py`), so results are computed identically either way. The full pipeline just adds sweeping, batching, and cluster submission on top.

### Prerequisites

Both pipelines expect the `CI` environment used throughout this repo. Set it up once, then activate it every time before running either pipeline:

```bash
module load python
conda create -n CI python=3.11   # only needed the first time
conda activate CI
pip install -r requirements.txt
```

Don't run the scripts in some other environment. `run_all.sh` / `array_job.sh` assume `conda activate CI` succeeds, and the exact package versions in `requirements.txt` are what the confidence-interval methods (bootstrap, `numba`-jitted stratified resampling, etc.) were validated against.

### Input data format

Both pipelines read a single CSV given by `relative_data_path` in your config. As the name indicates, the data path is relative to the root of the repository. Regarding the format, the data must have a `subtask` column (matching the `task` you set in your config) and an `alg_name` column identifying which algorithm produced that row. Beyond that, the required columns depend on whether you're evaluating a classification or a segmentation metric:

**Classification** — one row per sample, with the model's raw per-class scores and the true label:

| subtask | alg_name | logits | target |
|---|---|---|---|
| task1 | algoA | "[0.05, 0.90, 0.05]" | 1 |
| task1 | algoA | "[0.20, 0.30, 0.50]" | 0 |
| task1 | algoB | "[0.10, 0.10, 0.80]" | 2 |

- `logits`: a string-encoded list of per-class scores (e.g. softmax outputs or raw logits), one entry per class. Note: the current loader parses this with `eval(...)`, so the CSV must be trusted (do not run on untrusted input).
- `target`: the true class index (integer).
- The number of classes is inferred from the length of `logits`; every row for a given `(subtask, alg_name)` must have the same length. Rows for which the list in `logits` has a different length will be discarded during the processing.

**Segmentation** — one row per sample, with the metric value already computed for that sample:

| subtask | alg_name | score | value |
|---|---|---|---|
| task1 | algoA | dsc | 0.87 |
| task1 | algoA | hd | 3.20 |
| task1 | algoB | dsc | 0.81 |

- `value`: the per-sample metric value (e.g. that sample's Dice score).
- `score` (optional): if your file stores several metrics stacked together (as in the example above), include a `score` column and the pipeline will filter to the rows matching `metric` in your config.

In both cases, rows with `NaN` in the relevant column(s) are dropped automatically, and a `(subtask, alg_name)` pair needs **at least 50 valid rows** to be evaluated — pairs with fewer are skipped with a warning rather than failing the run.

---

### Option A — Run a single experiment

Use this when you just want the coverage for one metric on one task (optionally, one algorithm).

0. Make sure the `CI` environment is activated (`conda activate CI` — see Prerequisites above).

1. Copy one of the example configs and fill it in:
   - `src/cfg/single_experiment_example_classif.yaml` for classification metrics (`accuracy`, `f1_score`, `auc`, ...)
   - `src/cfg/single_experiment_example_segm.yaml` for segmentation metrics (`dsc`, `hd`, ...)

   At minimum, set `task` to a value that matches a `subtask` in your data file, and `metric` to the metric you want to evaluate. Leave `algo: null` to evaluate every algorithm found for that task, or set it to a specific algorithm name.

2. Run it from the repo root:

   ```bash
   python src/run_single.py --config src/cfg/my_experiment.yaml
   ```

3. The script prints a coverage table straight to the terminal once it's done, and writes the same raw/aggregated CSVs to `relative_output_dir` that the full pipeline produces — so a single-experiment result can later be picked up by `merge_dataframes.py` if you want to fold it into a larger results directory.

No SLURM, no Hydra multirun, no instance lists to generate first — everything needed is in the one config file.

---

### Option B — Run the full sweep pipeline (SLURM)

Use this to reproduce the paper's experiments: every metric × average/summary_stat pair, every ablation, every task and algorithm in your dataset, submitted as SLURM array jobs.

0. Make sure the `CI` environment is activated (`conda activate CI` — see Prerequisites above). `array_job.sh` re-activates it on every SLURM array task, but the submitting shell (where you run `run_all.sh`) needs it active too, since it calls `python src/utils/extract_sweep.py` and `python src/utils/build_task_list.py` directly before submitting anything.

1. Pick (or write) a top-level config under `src/cfg/<domain>/config_<domain>.yaml` (e.g. `src/cfg/classif/config_classif.yaml`), which points to:
   - a `default_<domain>.yaml` with the shared parameters (kernel, sample sizes, CI methods, ...)
   - a `sweep/*.yaml` file listing the (metric, average/summary_stat) pairs to sweep over
   - an `ablations/` folder with one config per named experiment (e.g. `epanechnikov_adaptive`, `gaussian_scott`)

2. Submit the whole sweep:

   ```bash
   ./run_all.sh classif/config_classif
   ```

   This will, in order:
   - generate the list of (task, algorithm) instances for every metric in the sweep (`extract_df_and_make_instance_list.py`)
   - build one task list per named experiment, crossing the sweep's (metric, average/summary_stat) pairs with every (task, algorithm) instance (`build_task_list.py`)
   - submit one SLURM array job per experiment (`array_job.sh`), each array task running `src/run.py` with the corresponding Hydra overrides

3. Once the jobs finish, fuse the per-(task, algorithm) result files into one CSV per (metric, average/summary_stat):

   ```bash
   python src/utils/merge_dataframes.py \
     --results_dir ../results_classif \
     --sweep_file src/cfg/classif/sweep/classif_all_pairs.yaml \
     --task_type classif \
     --delete_files
   ```

   (use `--task_type segm` and the matching sweep file for segmentation results).

---