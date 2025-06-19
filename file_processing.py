import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

input_folder = "/lustre/fswork/projects/rech/qhn/ube47qn/Code/results"
output_folder = "/lustre/fswork/projects/rech/qhn/ube47qn/Code/results_clean"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Group files by base name without final "_n"
file_groups = defaultdict(list)
pattern = re.compile(r"^(results_.+)_\d+\.csv$")

# Scan and group files
for fname in tqdm(os.listdir(input_folder)):
    if not fname.endswith(".csv"):
        continue
    match = pattern.match(fname)
    if match:
        base_name = match.group(1)
        file_groups[base_name].append(fname)

# Process each group
for base_name, files in tqdm(file_groups.items()):
    output_path = os.path.join(output_folder, base_name + ".csv")
    if os.path.exists(output_path):
        continue
    dfs = []
    for fname in files:
        fpath = os.path.join(input_folder, fname)
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
    if dfs:
        combined = pd.concat(dfs).drop_duplicates()
        if combined.shape[0]!=90000:
            print(output_path)
        combined.to_csv(output_path, index=False)

output_folder = "/lustre/fswork/projects/rech/qhn/ube47qn/Code/results_clean"

files = os.listdir(output_folder)

with open("list.txt", "w") as f:
    for file in tqdm(files):
        file_path = os.path.join(output_folder, file)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if df.shape[0]!=90000:
                print(df.shape[0], file)
                f.write(f"{file}\n")
        except:
            print(file)
            f.write(f"{file}\n")