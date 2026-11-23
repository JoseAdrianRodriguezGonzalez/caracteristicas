import pandas as pd
import numpy as np
df = pd.read_csv('features_dataset.csv')
niveles_submuestreo = ['20', '30', '40', '50']
for nivel in niveles_submuestreo:
    dx_col = f'dx_frac{nivel}'
    dy_col = f'dy_frac{nivel}'
    dz_col = f'dz_frac{nivel}'
    volumen_col = f'volumen_frac{nivel}'
    df[f'compacidad_frac{nivel}'] = df[volumen_col] / (df[dx_col] * df[dy_col] * df[dz_col])
columnas_volumen = [col for col in df.columns if 'volumen' in col]
df = df.drop(columns=columnas_volumen)
df.to_csv('dataset_sin_volumen_con_compacidad.csv', index=False)
print("\n¡Dataset guardado exitosamente!")