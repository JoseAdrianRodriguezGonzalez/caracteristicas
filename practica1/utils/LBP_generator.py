import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
from LBP import LBP
def calculate_features_lbp(images_path):
    features=['media','varianza','correlacion','contraste','homogeneidad']
    dataset={feature:[]  for feature in features}
    for sizes in range(1,8):
        print(f"Calculando características para ventana {sizes}...")
        lbp=LBP(images_path)
        lbp.compute_lbp()
        lbp.media(sizes)
        lbp.varianza(sizes)
        lbp.correlacion(sizes)
        lbp.contraste(sizes)
        lbp.homogeneidad(sizes)
        print(f"Calculando la media...")
        dataset['media'].append(lbp.media_images)
        print(f"Calculando la varianza...")
        dataset['varianza'].append(lbp.varianza_images)
        print(f"Calculando la correlacion...")
        dataset['correlacion'].append(lbp.correlacion_images)
        print(f"Calculando el contraste...")
        dataset['contraste'].append(lbp.contraste_images)
        print(f"Calculando la homogeneidad...")
        dataset['homogeneidad'].append(lbp.homogeneidad_images)
    return dataset 
