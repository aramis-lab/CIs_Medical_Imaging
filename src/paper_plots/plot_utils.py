metric_labels = {
    'dsc': 'DSC',
    'iou': 'IoU',
    'nsd': 'NSD',
    'boundary_iou': 'Boundary IoU',
    'cldice': 'clDice',
    'assd': 'ASSD',
    'masd' : 'MASD',
    'hd': 'HD',
    'hd_perc': 'HD95',
    'balanced_accuracy': 'Balanced Accuracy',
    'ap': 'AP',
    'auc': 'AUC',
    'f1_score': 'F1 Score',
    'accuracy': 'Accuracy'
}

stat_labels = {
    'mean': 'Mean',
    'median': 'Median',
    'std': 'Standard Deviation',
    'trimmed_mean': 'Trimmed Mean',
    'iqr_length': 'IQR Length'
}

method_labels = {
    "basic": "Basic",
    "percentile": "Percentile",
    "bca": "BCa",
    "delong": "DeLong",
    "logit_transform": "Logit Transform",
    "wilson": "Wilson",
    "agresti_coull" : "Agresti-Coull",
    "exact" : "Exact \n(Clopper-Pearson)",
    "wald" : 'Wald',
    "param_t" : "Parametric t",
    "param_z" : "Parametric z"
}

method_colors = {
    "basic": "#0077B6",
    "percentile": "#B39DDB", 
    "bca" : "#F4A261",
    "wilson" : "#DFCF3E", 
    "agresti_coull" : "#5D9336", 
    "exact" : "#DB4ADB", 
    "wald" : "#E07A5F",
    "param_t" : "#999999", 
    "param_z" : "#A7C7E7"}