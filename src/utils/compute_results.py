import pandas as pd
import os

def compute_results(input_dir, output_dir, metrics, summary_stats, tasks, algos):

    for metric in metrics:
        for summary_stat in summary_stats:
            results = pd.DataFrame()
            for task in tasks:
                for algo in algos:
                    if os.path.exists(os.path.join(input_dir, f"results_{metric}_{summary_stat}_{task}_{algo}.csv")):

                        results_df = pd.read_csv(os.path.join(input_dir, f"results_{metric}_{summary_stat}_{task}_{algo}.csv"), low_memory=False)

                        n_list = results_df["n"].unique()

                        columns_to_drop = ["sample_index"] + [c for c in results_df.columns if "bound" in c]
                        results_df = results_df.drop(columns=columns_to_drop)
                        
                        for n in n_list:
                            df_n = results_df[results_df["n"]==n]
                            mean_df = df_n.mean(numeric_only=True)
                            for col in df_n.columns:
                                if col not in mean_df:
                                    mean_df[col] = df_n[col].iloc[0]
                            results = pd.concat([results, mean_df.to_frame().T], ignore_index=True)
            results.to_csv(os.path.join(output_dir, f"aggregated_results_{metric}_{summary_stat}"))