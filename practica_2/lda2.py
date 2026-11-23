
from lib.LDA import LDA_Model
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Cargar rutas
directories_train = sorted([os.path.join("features/", f) 
    for f in os.listdir("features") if f.endswith(".csv") and "train" in f])[2:4]
directories_val = sorted([os.path.join("features/", f) 
    for f in os.listdir("features") if f.endswith(".csv") and "val" in f])[2:4]

def models():
    for train, validation in zip(directories_train, directories_val):
        print(f"\n[INFO] Procesando dataset: {train}-{validation}")

        df_train = pd.read_csv(train)
        df_val = pd.read_csv(validation)
        df = pd.concat([df_train, df_val], ignore_index=True)

        scaler = StandardScaler()
        X = df.drop(columns=["label"])
        X_scaled = scaler.fit_transform(X)
        y = df["label"]
        pca=PCA(n_components=0.9)
        X_pca=pca.fit_transform(X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=0.8, stratify=y, random_state=42
        )

        file_name = train[9:-4]+"pca"

        # ======== LDA ========
        lda = LDA_Model(solver="lsqr")
        lda.train(X_train, y_train)
        lda.evaluate(X_test, y_test, file_name)

        metrics = lda.metrics
        print(f"[RESULT] [LDA] -> {metrics['accuracy']:.3f} -> {metrics}")
        save_logs_ml(metrics, f"logs/LDA{file_name}.txt")
        print(f"[INFO] log almacenado en LDA{file_name}")
        plot_learning_curve(lda.model, X_pca, y, model_name="LDA" + file_name)

models()
