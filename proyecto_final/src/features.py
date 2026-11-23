from skimage.feature import hog 
import os 
import numpy as np
import pandas as pd 
from src.image_preprocessing import resize_image,load_image_from_txt
from scipy.stats import kurtosis,skew

def mean(vector_features):
    return np.mean(vector_features)
def std(vector_features):
    return np.std(vector_features)
def max(vector_features):
    return np.max(vector_features)
def min(vector_features):
    return np.min(vector_features)
def median(vector_features):
    return np.median(vector_features)
def quartil_1 (vector_features):
    return np.quantile(vector_features,.25)
def quartil_3(vector_features):
    return np.quantile(vector_features,.75)
def skew_f(vector_features):
    sk= skew(vector_features)
    return 0.0 if np.isnan(sk) else sk 
def kurtosis_f(vector_features):
    k =kurtosis(vector_features)
    return 0.0 if np.isnan(k) else k
def energia(vector_features):
    return np.sum(vector_features**2)
def features_calculation(vector_features,h):
    features={
        "mean":mean(vector_features),
        "std":std(vector_features),
        "max":max(vector_features),
        "min":min(vector_features),
        "median":median(vector_features),
        "quartil_1":quartil_1(vector_features),
    "quartil_3":quartil_3(vector_features),
        "skew":skew_f(vector_features),
        "kurtosis":kurtosis_f(vector_features),
        "energia":energia(vector_features),
    } 
    caracteristicas_41=h[:41]
    for idx,caracteristica in enumerate(caracteristicas_41):
        features[f"HoG_{idx}"]=caracteristica
    return features

def read_from_txt(root_path:str):
    labels=[]
    features=[]
    for dirs in os.listdir(root_path):
        if not os.path.isdir(os.path.join(root_path,dirs)):
            continue
        print(f"Leyendo la clase {dirs}")
        for file in os.listdir(os.path.join(root_path,dirs)):
            with open(os.path.join(root_path,dirs,file),"r",encoding="utf-8") as file:
                line=file.readlines()
            for idx,image in enumerate(line):
                labels.append(dirs)
                matriz=resize_image(image)
                frame_norm=matriz/ matriz.max() if matriz.max()>0 else matriz
                vector_features=hog(frame_norm,pixels_per_cell=(8,8),cells_per_block=(1,1),feature_vector=True)
                HoG_fe=hog(matriz,pixels_per_cell=(8,8),cells_per_block=(1,1),orientations=9,feature_vector=True)
                caracteristicas_calculadas=features_calculation(vector_features,HoG_fe)
                features.append(caracteristicas_calculadas)

    return features, labels
def read_image_from_txt_features(root_path:str):
    labels=[]
    features=[]
    for dirs in os.listdir(root_path):
        if not os.path.isdir(os.path.join(root_path,dirs)):
            continue
        print(f"Leyendo la clase {dirs}")
        imagenes=[load_image_from_txt(os.path.join(root_path,dirs,file)) for file in (sorted(os.listdir(os.path.join(root_path,dirs))))]
        for imagen in imagenes:
            labels.append(dirs)
            frame_norm=imagen/ imagen.max() if imagen.max()>0 else imagen
            vector_features=hog(frame_norm,pixels_per_cell=(8,8),cells_per_block=(1,1),feature_vector=True)
            HoG_fe=hog(imagen,pixels_per_cell=(8,8),cells_per_block=(1,1),orientations=9,feature_vector=True)
            caracteristicas_calculadas=features_calculation(vector_features,HoG_fe)
            features.append(caracteristicas_calculadas)
    return features, labels
