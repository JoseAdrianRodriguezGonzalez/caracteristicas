# Notas
En este directorio se tienen los siguientes archivos.
 ada_plopt.py
 adaboost.py
 features_dataset_completo.csv
 gradient.py
 gradient_plot.py
 observaciones.py
 package.py
 preproccesing.py
 random_forest.py
 random_plot.py
 readme.md
 stat_test.py

Los archivos que tiene adaboost, random_forest y gradient, buscan los hiperpárametros. Aquellos con los nombres random_plot,ada_plot y gradient_plot, son aquellos que permiten graficar la curva de aprendiza y la matriz de confusión.
El archivo `observaciones`, es el que hace los calculos de las divergencias de Kullback Leibler y Jensen Shannon
El archivo stat_test, contiene el cálculo de estadísticas descriptivas y las pruebas de normalidad. package contiene funciones importantes que son usadas en los demás archivos.

## Versión de python
Se usa la versión de python 3.13.5
Se usan los siguientes paquetes
scikit-learn
numpy
statsmodels
scipy
pandas
matplotlib
optuna
os
seaborn