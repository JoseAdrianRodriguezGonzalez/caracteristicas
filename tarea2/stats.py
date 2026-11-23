import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis,shapiro,median_abs_deviation
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("features_dataset_completo.csv")
df_features=dataset.drop(columns="category")
summary_list=[]
for feature in df_features.columns:
    data=df_features[feature].values

    #estadisticos
    mean_val=np.mean(data)
    var_val=np.var(data,ddof=1)
    skew_val=skew(data)
    kurt_val=kurtosis(data)
    median=np.median(data)
    p05=np.percentile(data,q=5)
    p25=np.percentile(data,q=25)
    p5=np.percentile(data,q=50)
    p75=np.percentile(data,q=75)
    p95=np.percentile(data,q=95)
    medina_abs=median_abs_deviation(data)
    IQR=p75-p25
    cv=medina_abs/mean_val
    stat,p_value=shapiro(data)
    summary_list.append({
        'feature': feature,
        'mean': mean_val,
        'variance': var_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'percentil 5':p05,
        'percentil 25':p25,
        'percentil 50':p5,
        'percentil 75':p75,
        'percentil 95':p95,
        'mediana absoluta':medina_abs,
        'IQR':IQR,
        'CV':cv,
        'shapiro_stat': stat,
        'shapiro_pvalue': p_value,
        'normal': p_value > 0.05
    })
    plt.figure(figsize=(6,4))
    sns.histplot(data, kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {feature}\nShapiro p={p_value:.3f}, normal={p_value>0.05}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
df_summary=pd.DataFrame(summary_list)
pd.set_option('display.float_format',lambda x:'%.4f'%x)
df_summary.to_csv("completo_stats.csv")
print(df_summary)
