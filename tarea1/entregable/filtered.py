##Primer archivo a observar, proceso de filtrado

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df=pd.read_csv('./data/silla/2025_08_25_17_06_40/capture_filtered_3.csv',header=None)
df.columns=['x','y','z']
y_min=-0.37
y_max=0.5
x_min=-0.39
x_max=0.4
z_min=-0.85
z_max=-0.2
df_filtrado = df[
                #(df['x'] >= x_min) & (df['x'] <= x_max)&
                #(df['z'] >= z_min) & (df['z'] <= z_max)&
                (df['y'] >= y_min) & (df['y'] <= y_max)]
                
                

def previsualizar(imagen):
    fig=plt.figure(figsize=(8,6))
    ax =fig.add_subplot(111,projection='3d')

    ax.scatter(imagen['x'],imagen['y'],imagen['z'],s=5,c='blue',alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nube de puntos')

    plt.show()
#df_filtrado.to_csv('./data_good/silla/capture_filtered_11.csv',index=False)
previsualizar(df_filtrado)