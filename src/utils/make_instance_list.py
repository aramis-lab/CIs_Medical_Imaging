from extract_df_and_instances import get_benchmark_instances
from omegaconf import OmegaConf
import os


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cfg_path = "../cfg/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    instances = get_benchmark_instances(BASE_DIR, cfg)

    with open("../benchmark_list.txt", "w") as f:
        for task, algo in instances:
            f.write(f"{task} {algo}\n")