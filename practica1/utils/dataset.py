from SDH_Generator import flattenized,calculate_features,summarized,summarized_Xy,summarized_glcm

from GLCM_generator import calculate_glcm_features
from LBP_generator import calculate_features_lbp
import pandas as pd 

def generate_dataset_csv(images_path):
    print("=========Calculo de caracteristicas=====")
    dataset=calculate_features(images_path)
    features=list(dataset.keys())
    n_sizes=len(dataset[features[0]])
    n_total=len(features)*n_sizes
    expanded_feature=[]
    for i in range(1,n_sizes+1):
        for feature in features:
            expanded_feature.append(f"{feature}_{i}")
    #flatenized
    print("Generando dataset completo (pixel-level, 84 características)...")
    X,y=flattenized(dataset)
    if X.shape[1] != n_total:
        print(f" Advertencia: se esperaban {n_total} columnas, pero se obtuvieron {X.shape[1]}")

    df=pd.DataFrame(X,columns=expanded_feature)
    df['label']=y
    print("generacion de csv completo")
    df.to_csv('../out/dataset_full.csv',index=False)
    #summarized
    print("Generando dataset resumido (imagen-level, 84 características)...")
 
    mean_df=summarized(dataset)
    X_sum,y_sum=summarized_Xy(mean_df)
    df_sum=pd.DataFrame(X_sum,columns=expanded_feature)
    df_sum['label']=y_sum
    print("generacion de csv promedio")
    df_sum.to_csv('../out/dataset_sum.csv',index=False)
    print("archivos generados correctamente")
def generate_dataset_GLCM_csv(images_path):
    print("=========Calculo de caracteristicas=====")
    dataset=calculate_glcm_features(images_path)
    features=list(dataset.keys())
    n_sizes=len(dataset[features[0]])
    n_total=len(features)*n_sizes
    expanded_feature=[]
    for i in range(1,n_sizes+1):
        for feature in features:
            expanded_feature.append(f"{feature}_{i}")
    rows=summarized_glcm(dataset)
    X,y=summarized_Xy(rows)
    df=pd.DataFrame(X,columns=expanded_feature)
    df['label']=y
    print("generacion de csv promedio")
    df.to_csv('../out/dataset_glcm.csv',index=False)
    print("archivos generados correctamente")


def generate_dataset_lbp(images_path):
    print("=========Calculo de caracteristicas=====")
    dataset=calculate_features_lbp(images_path)
    features=list(dataset.keys())
    n_sizes=len(dataset[features[0]])
    n_total=len(features)*n_sizes
    expanded_feature=[]
    for i in range(1,n_sizes+1):
        for feature in features:
            expanded_feature.append(f"{feature}_{i}")
    print(len(dataset['media']))
    #flatenized
    print("Generando dataset completo (pixel-level, 84 características)...")
    X,y=flattenized(dataset,10)
    if X.shape[1] != n_total:
        print(f" Advertencia: se esperaban {n_total} columnas, pero se obtuvieron {X.shape[1]}")

    df=pd.DataFrame(X,columns=expanded_feature)
    df['label']=y
    print("generacion de csv completo")
    df.to_csv('../out/dataset_full_lbp.csv',index=False)
    #summarized
    print("Generando dataset resumido (imagen-level, 84 características)...")
 
    mean_df=summarized(dataset)
    X_sum,y_sum=summarized_Xy(mean_df,10)
    df_sum=pd.DataFrame(X_sum,columns=expanded_feature)
    df_sum['label']=y_sum
    print("generacion de csv promedio")
    df_sum.to_csv('../out/dataset_lbp.csv',index=False)
    print("archivos generados correctamente")