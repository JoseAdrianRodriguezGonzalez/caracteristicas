import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
from histograms import Histogram
def calculate_features(images_path):
    features=['media','varianza','correlacion','contraste','homogeneidad','cluster_sombra','cluster_prominencia']
    dataset={feature:[]  for feature in features}
    for sizes in range(1,13):
        print(f"Calculando características para ventana {sizes}...")
        SDH=Histogram(images_path,num_size=sizes)
        SDH.procesar_todo()
        print(f"Calculando la media...")
        dataset['media']=SDH.media_images
        print(f"Calculando la varianza...")
        dataset['varianza']=SDH.desviacion_images
        print(f"Calculando la correlacion...")
        dataset['correlacion']=SDH.correlacion_images
        print(f"Calculando el contraste...")
        dataset['contraste']=SDH.contraste_images
        print(f"Calculando la homogeneidad...")
        dataset['homogeneidad']=SDH.homogeneidad_images
        print(f"Calculando el cluster de sombra...")
        dataset['cluster_sombra']=SDH.sombra_images
        print(f"Calculando el cluster de prominencia...")
        dataset['cluster_prominencia']=SDH.prominencia_images
    return dataset 
def flattenized(all_features, n_per_class=50):
    print("proceso de flatenizacion")
#    print(all_features)
    features_name=list(all_features.keys())
    n_sizes=len(all_features[features_name[0]])
    n_images=len(all_features[features_name[0]][0])
    X_list = []
    y_list = []
    for img_idx in range(n_images):
        label = img_idx // n_per_class
        pixel_features_per_window = []
        for size_idx in range(n_sizes):
            pixel_features_list = []
            for feature in features_name:
                mat = all_features[feature][size_idx][img_idx]
                pixel_features_list.append(mat.ravel())
            pixel_features = np.stack(pixel_features_list, axis=1)  # shape (n_pixeles, 7)
            pixel_features_per_window.append(pixel_features)
        pixel_feature_concat=np.concatenate(pixel_features_per_window,axis=1)
        X_list.append(pixel_feature_concat)
        y_list.append(np.full(pixel_feature_concat.shape[0], label, dtype=np.int32))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y
def summarized(all_features):
    print("Proceso de promediar imagenes")
    features_name=list(all_features.keys())
    df_sumarized=[]
    n_sizes = len(all_features[features_name[0]])
    n_images = len(all_features[features_name[0]][0])
    for image in range(n_images):
        feature_vector=[]
        for size in range(n_sizes):
            for feature in features_name:
                img=all_features[feature][size][image]
                feature_vector.append(np.mean(img))
        df_sumarized.append(feature_vector)
    return np.array(df_sumarized,dtype=np.float32)


def summarized_glcm(all_features):
    print("Proceso de promediar imagenes")
    features_name=list(all_features.keys())
    df_sumarized=[]
    n_sizes = len(all_features[features_name[0]])
    n_images = len(all_features[features_name[0]][0])
    for image in range(n_images):
        feature_vector=[]
        for size in range(n_sizes):
            for feature in features_name:
                img=all_features[feature][size][image]
                feature_vector.append(img)
        df_sumarized.append(feature_vector)
    return np.array(df_sumarized,dtype=np.float32)

def summarized_Xy(df_summarized,n_per_samples=50):
    print("proceso de sumarizacion x,y de promedios")
    X=np.array(df_summarized,dtype=np.float32)
    nrow=X.shape[0]
    y = np.array([row // n_per_samples % 4 for row in range(nrow)], dtype=np.int32)
    return X,y
