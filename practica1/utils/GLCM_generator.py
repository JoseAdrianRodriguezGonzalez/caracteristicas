import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
from GLCM import GLCM
def calculate_glcm_features(images_path):
    features = ['energia', 'contraste', 'correlacion', 'homogeneidad', 'idf', 'entropia', 'varianza']
    dataset = {feature: [] for feature in features}

    distancias = [1, 3, 7]
    angulos = [0, 45, 90, 135]

    for d in distancias:
        for ang in angulos:
            print(f"Calculando características para distancia={d}, ángulo={ang}°...")
            glcm = GLCM(images_path,2*d+1)
            glcm.compute_all(distancia=d,angulo=ang)

            dataset['energia'].append(glcm.Energia)
            dataset['contraste'].append(glcm.Contraste)
            dataset['correlacion'].append(glcm.Correlacion)
            dataset['homogeneidad'].append(glcm.Homogeneidad)
            dataset['idf'].append(glcm.IDF_)
            dataset['entropia'].append(glcm.Entropia)
            dataset['varianza'].append(glcm.Varianza)

    return dataset
