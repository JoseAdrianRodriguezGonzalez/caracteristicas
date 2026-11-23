import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

df = pd.read_csv("features_dataset_completo.csv").drop(columns=["category","fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20"])
X = df.values
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
features = df.columns
n = len(features)
js_matrix = np.zeros((n, n))
kl_matrix = np.zeros((n, n))
with open("diagnostico_features.txt", "w") as f:
    f.write("Resumen de VIF por feature:\n")
    for idx, row in vif_data.iterrows():
        f.write(f"{row['feature']}: VIF={row['VIF']:.2e}\n")
    
    f.write("\n KL y JS divergencia entre features:\n")
    
    epsilon = 1e-10
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            hist1, bins = np.histogram(df[f1], bins=30, density=True)
            hist2, _ = np.histogram(df[f2], bins=bins, density=True)

            hist1 += epsilon
            hist2 += epsilon

            kl_div = entropy(hist1, hist2)
            kl_matrix[i,j]=kl_div
            hist1_norm = hist1 / np.sum(hist1)
            hist2_norm = hist2 / np.sum(hist2)
            js_div = jensenshannon(hist1_norm, hist2_norm)
            js_matrix[i,j]=js_div
            f.write(f"{f1}-{f2}: KL={kl_div:.4f}, JS={js_div:.4f}\n")

print("Archivo 'diagnostico_features.txt' generado con éxito.")
avg_kl = [np.nanmean(np.delete(kl_matrix[i,:], i)) for i in range(n)]
avg_js = [np.nanmean(np.delete(js_matrix[i,:], i)) for i in range(n)]

df_summary = pd.DataFrame({
    "feature": features,
    "avg_KL": avg_kl,
    "avg_JS": avg_js
})

print(df_summary)


print("="*100)
print("Ordenar datos")
vif_sorted = vif_data.sort_values(by="VIF", ascending=False)
print("Ranking por VIF (mayor a menor):")
print(vif_sorted)

# Ordenar avg_JS de menor a mayor (features más redundantes primero)
js_sorted = df_summary.sort_values(by="avg_JS", ascending=True)
print("Ranking por avg_JS (menor a mayor):")
print(js_sorted)

# Ordenar avg_KL de menor a mayor (features más redundantes/distribuciones similares)
kl_sorted = df_summary.sort_values(by="avg_KL", ascending=True)
print("Ranking por avg_KL (menor a mayor):")
print(kl_sorted)

df_summary["redundancy_score"] = df_summary["avg_JS"] / df_summary["avg_KL"] * vif_data["VIF"]
redundancy_sorted = df_summary.sort_values(by="redundancy_score", ascending=False)
print("Ranking por score de redundancia:")
print(redundancy_sorted)