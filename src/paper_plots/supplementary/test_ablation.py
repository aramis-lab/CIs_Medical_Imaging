import os 
import pandas as pd 

from .make_correction_fdr import results_ablation_df

def get_leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from get_leaves(value)
    else:
        yield obj

for kde, res_kde in results_ablation_df.groupby("kde"):

    for test, res_test in res_kde.groupby("test"):

        pvalues_all = []

        for pvalue_dict in res_test["pvalues_corrected"]:
            print(pvalue_dict)

