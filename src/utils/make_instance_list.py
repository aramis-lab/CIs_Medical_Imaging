# make_benchmark_list.py
from extract_df_and_instances import get_benchmark_instances
from CIs_Medical_Imaging.src.run import BASE_DIR
from omegaconf import OmegaConf

cfg_path = "cfg/config.yaml"
cfg = OmegaConf.load(cfg_path)
instances = get_benchmark_instances(BASE_DIR, cfg)

with open("../benchmark_list.txt", "w") as f:
    for task, algo in instances:
        f.write(f"{task} {algo}\n")