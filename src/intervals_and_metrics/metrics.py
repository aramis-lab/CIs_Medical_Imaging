import numpy as np
import pandas as pd
import os

def compute_metrics(input_dir, output_dir, metrics, summary_stats, tasks, algos):

    results_dict = {}
    for metric in metrics:
        for summary_stat in summary_stats:
            results = pd.DataFrame()
            for task in tasks:
                for algo in algos:
                    if os.path.exists(os.path.join(input_dir, f"results_{metric}_{summary_stat}_{task}_{algo}.csv")):

                        results_df = pd.read_csv(os.path.join(input_dir, f"results_{metric}_{summary_stat}_{task}_{algo}.csv"))

                        n_list = results_df["n"].unique()

                        columns_to_drop = ["sample_index"] + [c for c in results_df.columns if "bound" in c]
                        results_df = results_df.drop(columns=columns_to_drop)
                        
                        results = pd.DataFrame(columns=results_df.columns)
                        for n in n_list:
                            df_n = results_df[results_df["n"]==n].drop(columns = columns_to_drop)
                            results = pd.concat([results, df_n.mean()])
            results.to_csv(os.path.join(output_dir, f"aggregated_results_{metric}_{summary_stat}"))

    return results_dict