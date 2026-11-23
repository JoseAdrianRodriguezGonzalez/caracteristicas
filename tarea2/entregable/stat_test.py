import pandas as pd
import numpy as np
from scipy.stats import (
    skew, kurtosis, shapiro, median_abs_deviation,
    anderson, cramervonmises, normaltest, combine_pvalues
)
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("features_dataset_completo.csv")
df_features = dataset.drop(columns="category")

summary_list = []
for feature in df_features.columns:
    data=df_features[feature].values
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    skew_val = skew(data)
    kurt_val = kurtosis(data)
    median = np.median(data)
    p05 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)
    med_abs = median_abs_deviation(data)
    IQR = p75 - p25
    cv = med_abs / mean_val if mean_val != 0 else np.nan
    sh_stat,sh_p=shapiro(data)

    #Anderson darlin
    ad_res=anderson(data)
    ad_stat=ad_res.statistic
    #Nivel critico
    ad_crit=ad_res.critical_values[2]
    ad_normal=ad_stat<ad_crit


     # Cramér–von Mises
    cvm_res = cramervonmises(data, 'norm')
    cvm_stat, cvm_p = cvm_res.statistic, cvm_res.pvalue

    # D’Agostino-Pearson
    dag_stat, dag_p = normaltest(data)
    fisher_chi2, fisher_p = combine_pvalues([sh_p, cvm_p, dag_p], method='fisher')

    summary_list.append({
        'feature': feature,
        'mean': mean_val,
        'variance': var_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'percentil 5': p05,
        'percentil 25': p25,
        'percentil 50': p50,
        'percentil 75': p75,
        'percentil 95': p95,
        'mediana absoluta': med_abs,
        'IQR': IQR,
        'CV': cv,

        # Tests individuales
        'shapiro_stat': sh_stat,
        'shapiro_p': sh_p,
        'dagostino_stat': dag_stat,
        'dagostino_p': dag_p,
        'cvm_stat': cvm_stat,
        'cvm_p': cvm_p,
        'anderson_stat': ad_stat,
        'anderson_crit(5%)': ad_crit,
        'anderson_normal': ad_normal,

        # Combinación
        'fisher_chi2': fisher_chi2,
        'fisher_p': fisher_p,

        # Clasificación
        'normal_shapiro': sh_p > 0.05,
        'normal_dagostino': dag_p > 0.05,
        'normal_cvm': cvm_p > 0.05,
        'normal_anderson': ad_normal,
        'normal_fisher': fisher_p > 0.05
    })
    if fisher_p<0.05:
        print(f"a la caracteristica {feature} Se recomienda transformar")
        if kurt_val>3:
            print(f"{feature} se recomeinda transformarla con Box cox")
        elif skew_val>1:
            print(f"{feature}, se recomienda log,sqrt y box")
        elif skew_val<-1:
            print(f"{feature}, se recomienda Yeo-Johnsohn o invertir")
df_summary = pd.DataFrame(summary_list)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_summary.to_csv("completo_stats_con_tests.csv", index=False)
print(df_summary)