
import numpy as np
import pandas as pd
import scipy.stats as stats


# Define kolmogorov Smirnov test functions for each distribution

def ks_norm(data, score): 
   
    mu, sigma= stats.norm.fit(data)
    
    result= stats.kstest(data, 'norm', args=(mu, sigma))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }

def ks_beta(data, score): 
    try:
        a, b, loc, scale = stats.beta.fit(np.clip(data, 1e-6, 1-1e-6), floc=0, fscale=1)
    except Exception as e:
        print(f"Beta fit failed for score {score}: {e}")
        return None  # or some default result
    
    result= stats.kstest(np.clip(data, 1e-6, 1-1e-6), 'beta', args=(a, b, loc, scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }
def ks_expon(data, score): 
    loc, scale = stats.expon.fit(data)
    
    result= stats.kstest(data, 'expon', args=(loc, scale))
    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }
def ks_logistic(data, score): 
   
    loc, scale = stats.logistic.fit(data)
    
    result= stats.kstest(data, 'logistic', args=(loc, scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }


def ks_lognorm(data, score): 
    shape, loc, scale = stats.lognorm.fit(np.clip(data, 1e-6, None ))
    
    result= stats.kstest(np.clip(data, 1e-6, None ), 'lognorm', args=(shape, loc, scale))
    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }



def ks_skewnorm(data, score): 
   
    a, loc, scale = stats.skewnorm.fit(data)
    
    result= stats.kstest(data, 'skewnorm', args=(a, loc, scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }

def ks_pareto(data, score): 
   
    b = stats.pareto.fit(data)
    
    result= stats.kstest(data, 'pareto', args=(b))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }



def ks_chi2(data, score): 
   
    df, loc , scale = stats.chi2.fit(data)
    result= stats.kstest(data, 'chi2', args=(df, loc , scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }



def ks_student(data, score): 
   
    
    df, loc , scale = stats.t.fit(data)

    result= stats.kstest(data, 't', args=(df, loc , scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }


def ks_gamma(data, score): 
   
    a, loc, scale= stats.gamma.fit(data)
    result= stats.kstest(data, 'gamma', args=(a, loc, scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }


def ks_weibull(data, score): 
   
    c, loc, scale= stats.weibull_min.fit(data)
    result= stats.kstest(data, 'weibull_min', args=(c, loc, scale))

    # Store the result in a DataFrame
    
    return {
        'Metric': score,
        'KS_Statistic / effet size': result.statistic,
        'P_Value': result.pvalue, 
        'Sample_Size': len(data), 
    }



# Get data for segmentation 

data_segm=pd.read_csv("../../données/segmentation/data_matrix_grandchallenge_all.csv", sep=";")


distributions_segm=  ["Normal", "Student", "Skewnorm", "Log-Normal", "Logistic", "Exponential","Chi2", "Beta", "Gamma", "Pareto", "Weibull"]
reject_df=pd.DataFrame(columns= ['Metric', "Normal", "Student", "Skewnorm", "Log-Normal", "Logistic", "Exponential", "Chi2", "Beta", "Gamma", "Pareto", "Weibull"])

effect_size_list=[]
rejected_list = [] 
metric_list=[]
test_list=[]
significance_list=[]
metrics = data_segm['score'].unique()
metrics_bounded=['nsd', 'cldice','dsc', 'iou', 'boundary_iou']
reject_df['Metric']=metrics

for test in distributions_segm:

    reject_list=[]
    for score in metrics:
       
        df=data_segm[data_segm['score']==score]
        algos=df['alg_name'].unique()
        score=df['score'].unique()[0]
        all_results = []
        count_total=0
        count_rejects=0
        for alg in algos:
            df_alg= df[df['alg_name']==alg]
            tasks = df_alg['subtask'].unique()
            
            for task in tasks:
                
                values = df_alg[df_alg['subtask'] == task]['value'].dropna()
                if len(values)<50:
                    if task not in ['Task02_Heart_L1', 'Task05_Prostate_L1', 'Task05_Prostate_L2', 'Task06_Lung_L1', 'Task09_Spleen_L1']:
                        print('skipped task', task )
                    continue
                
                count_total+=1
                if test =='Normal': 
                    result=ks_norm(values, score)
                
                elif test =='Beta':
                    if score in metrics_bounded:
                        result = ks_beta(values, score)
                        if result is None:
                            continue
                    else:
                        count_rejects=np.nan
                        continue
                elif test == 'Pareto':
                    result=ks_pareto(values, score)
                elif test == 'Chi2':
                    result=ks_chi2(values, score)
                elif test == 'Student':
                    result=ks_student(values, score)
                elif test == 'Gamma':
                    result=ks_gamma(values, score)
                elif test== 'Weibull':
                    result=ks_weibull(values, score)
                elif test == 'skewnorm':
                    result=ks_skewnorm(values, score)
                elif test == 'Log-Normal': 
                    result=ks_lognorm(values, score)
                elif test == 'Exponential':
                    result=ks_expon(values, score)
                else: 
                    result=ks_logistic(values, score)

                if result['P_Value']<=0.05:
                    count_rejects+=1

        reject_list.append(count_rejects/count_total)
       
    reject_df[test]=reject_list


ks_segm_df=reject_df.set_index("Metric")
ks_segm_df=np.round(ks_segm_df*100,1)

print(np.round(ks_segm_df))


# Get data classif

data_classif=pd.read_csv("../../données/classif/data_matrix_classification.csv")

def parse_vector_string(vec_str):
    # Remove curly braces and split by comma
    vec_str = vec_str.strip('{}')
    parts = vec_str.split(',')
    # Convert each part to float (strip whitespace just in case)
    return [float(x.strip()) for x in parts]


rejected_list = [] 

for test in ['Pareto', 'Skewnorm',  'Student', 'Normal', 'Logistic']:
 
    algos=data_classif['alg_name'].unique()
    count_total=0
    count_rejects=0
    for alg in algos:
        df_alg= data_classif[data_classif['alg_name']==alg]
        tasks = df_alg['subtask'].unique()
        
        for task in tasks:
            
            logits = df_alg[df_alg['subtask'] == task]['logits'].dropna()
            list_of_lists = logits.apply(parse_vector_string).tolist()
            max_len = max(len(vec) for vec in list_of_lists)
            list_of_lists = [vec for vec in list_of_lists if len(vec) == max_len]
            matrix_logits = np.array(list_of_lists)
            n_row, n_col=matrix_logits.shape
            count_total+=1
            if n_row<50:
                    print(f"not enough samples, ({n_row}), skipped task {task}")
                    continue
            for classes in range(n_col): 
               
                dist= matrix_logits[:,classes]
              
                if test =='Normal': 
                    result=ks_norm(dist, 'classif')            
                elif test == 'Pareto':
                    result=ks_pareto(dist, 'classif')
                elif test == 'Chi2':
                    result=ks_chi2(dist, 'classif')
                elif test == 'Student':
                    result=ks_student(dist, 'classif')
                elif test== 'Weibull':
                    result=ks_weibull(dist, 'classif')
                elif test == 'Skewnorm':
                    result=ks_skewnorm(dist, 'classif')
                else: 
                    result=ks_logistic(dist, 'classif')
                if result['P_Value']<=0.05:
                    count_rejects+=1
                    break
   
    rejected_list.append({'Distribution': test, 
                          'Proportion of reject':count_rejects/count_total })
    
ks_classif_df=pd.DataFrame(rejected_list)
print(ks_classif_df)