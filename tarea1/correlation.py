import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['figure.figsize'] = (16, 14) 
plt.rcParams['font.size'] = 10
df = pd.read_csv('dataset_sin_volumen_con_compacidad.csv')
df=df.drop(columns=['category'])
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(18, 16))
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'white', 'red'])
heatmap = sns.heatmap(correlation_matrix,mask=mask,annot=True, annot_kws={'size': 8},cmap=cmap,
center=0,square=True,fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlación de 40 Características\n', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('heatmap_correlacion.png', dpi=300, bbox_inches='tight')

plt.show()
print("\nCorrelaciones más fuertes (absolutas > 0.8):")
strong_correlations = correlation_matrix.unstack().sort_values(key=abs, ascending=False)
strong_correlations = strong_correlations[strong_correlations != 1.0] 
for idx, value in strong_correlations.items():
    if abs(value) > 0.8: 
        print(f"{idx[0]} - {idx[1]}: {value:.3f}")