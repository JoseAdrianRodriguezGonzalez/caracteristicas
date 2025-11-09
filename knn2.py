from lib.Knn import Knn
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import matplotlib.pyplot as plt

directories_train = sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "train" in f])
directories_val=sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "val" in f])

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
        y = df["label"]
        
        X_train, X_test, y_train, y_test =[],[],[],[]
        print(y.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, train_size=0.8, stratify=y, random_state=42
            )
        # ======== KNN MODELOS ========
        knn_configs = [
            {"model": "KNN_Fine", "n_neighbors": 3},
            {"model": "KNN_Medium", "n_neighbors": 5},
            {"model": "KNN_Coarse", "n_neighbors": 10},
        ]
        file_name=train[9:-4]
        print(file_name)
        for config in knn_configs:
            knn_model = Knn(config)
            print(f"[INFO] Entrenando {config['model']}")
            knn_model.train(X_train, y_train)
            knn_model.evaluate(X_test,y_test,"cnn/"+file_name)
            metrics = knn_model.metrics
            print(f"[RESULT] {config['model']} -> {metrics['accuracy']:.3f} -> {metrics}")
            save_logs_ml(metrics,f"logs/cnn/{knn_model.name+file_name}.txt")
            print(f"[INFO] log almacenado en {knn_model.name+file_name}")
            plot_learning_curve(knn_model.model, X_scaled, y, model_name="cnn/"+config['model']+file_name)
models()
