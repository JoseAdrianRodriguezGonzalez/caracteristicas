#Cuarto archivo Graficas de las variables
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
df = pd.read_csv("./dataset_ordenado.csv")
columnas = [cols for cols in df.columns if cols != 'category']
print(f"Número de características: {len(columnas)}")

grupos = []
for i in range(0, len(columnas), 3):
    grupo = columnas[i:i+3]
    if len(grupo) == 3:
        grupos.append(grupo)
print(f"Se crearán {len(grupos)} gráficas 3D")
for i, grupo in enumerate(grupos):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    categorias = df['category'].unique()
    colores = plt.cm.tab10(np.linspace(0, 1, len(categorias)))
    for j, categoria in enumerate(categorias):
        datos_categoria = df[df['category'] == categoria]
        ax.scatter(datos_categoria[grupo[0]], 
                  datos_categoria[grupo[1]], 
                  datos_categoria[grupo[2]], 
                  label=categoria, 
                  alpha=0.7, 
                  s=50,
                  c=[colores[j]])
    ax.set_xlabel(grupo[0], fontsize=12)
    ax.set_ylabel(grupo[1], fontsize=12)
    ax.set_zlabel(grupo[2], fontsize=12)
    ax.set_title(f'Gráfica {i+1}: {grupo[0]} vs {grupo[1]} vs {grupo[2]}', fontsize=14)
    ax.legend()
    plt.savefig(f'grafica_combinada_{i+1:02d}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

print("¡Todas las gráficas han sido creadas!")