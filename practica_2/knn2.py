from lib.Knn import Knn
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

directories_train = sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "train" in f])[2:4]
directories_val=sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "val" in f])[2:4]

def models():
    for train,validation in zip(directories_train,directories_val):
        print(f"\n[INFO] Procesando dataset: {train}-{validation}")
        df_train=pd.read_csv(train)
        print(df_train.shape)
        df_val=pd.read_csv(validation)
        print(df_val.shape)
        df=pd.concat([df_train,df_val],ignore_index=True)
        print(df.shape)
        scaler=StandardScaler()
        nan_cols=df.isna().sum()
        print(nan_cols[nan_cols>0])
        
        X = df.drop(columns=["label"])
        X_scaled=scaler.fit_transform(X)
       # pca=PCA(n_components=.90)
       # X_pca=pca.fit_transform(X_scaled)

        y = df["label"]
        lda=LDA()
        X_lda=lda.fit_transform(X_scaled,y)
        explained_ratios = lda.explained_variance_ratio_
        cumulative = np.cumsum(explained_ratios)
        n_comp = np.searchsorted(cumulative, 0.90) + 1
        print(f"[INFO] Componentes necesarios para 90% de 'varianza discriminante': {n_comp}")
        lda=LDA(n_components=n_comp)
        X_lda=lda.fit_transform(X_scaled,y)
        
        X_train, X_test, y_train, y_test =[],[],[],[]
        print(y.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(
                X_lda, y, train_size=0.8, stratify=y, random_state=42
            )
        # ======== KNN MODELOS ========
        knn_configs = [
            {"model": "KNN_Fine", "n_neighbors": 3},
            {"model": "KNN_Medium", "n_neighbors": 5},
            {"model": "KNN_Coarse", "n_neighbors": 10},
        ]
        file_name=train[9:-4]+"lda_"+str(n_comp)
        print(file_name)
        for config in knn_configs:
            knn_model = Knn(config)
            print(f"[INFO] Entrenando {config['model']}")
            knn_model.train(X_train, y_train)
            knn_model.evaluate(X_test,y_test,file_name)
            metrics = knn_model.metrics
            print(f"[RESULT] {config['model']} -> {metrics['accuracy']:.3f} -> {metrics}")
            save_logs_ml(metrics,f"logs/cnn/{knn_model.name+file_name}.txt")
            print(f"[INFO] log almacenado en {knn_model.name+file_name}")
            plot_learning_curve(knn_model.model, X_lda, y, model_name=config['model']+file_name)
models()
