#Tercer archivo: Llamada a la clase y almaenamiento y formacion de características
import os
import pandas as pd
from features import features

root_dir = './objetos/'
directions_list = [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files]
lista_objetos = ['carro','cloro','cono','munieco','silla','taza']

fractions = [0.2, 0.3, 0.4, 0.5]
base_feature = ['compacidad','dx','dy','dz','cx','cy','cz','curvature','border','fpfh']

# Crear nombres de columnas para cada submuestreo
feature_columns = []
for frac in fractions:
    feature_columns.extend([f"{col}_frac{int(frac*100)}" for col in base_feature])
feature_columns.append('category')

rows = []

for i, obj in enumerate(lista_objetos):
    for path in directions_list:
        if obj in path:
            cara = features(path)
            row = []
            for frac in fractions:
                pts_sub = cara.subsample(frac)
                compacidad = cara.feature_compacity(pts_sub)
                dim = cara.feature_dimensions(pts_sub)
                centro = cara.centroid(pts_sub)
                curva = cara.feature_curvature(pts_sub)
                borde = cara.feature_border(pts_sub)
                fpfh = cara.feature_fpfh(pts_sub)
                print(path,frac)
                row.extend([compacidad, dim[0], dim[1], dim[2], centro[0], centro[1], centro[2], curva, borde, fpfh])
            row.append(obj)  # categoría
            rows.append(row)

df = pd.DataFrame(rows, columns=feature_columns)
df.to_csv("features_dataset_completo.csv", index=False)
print("CSV con submuestreos creado")
