
from lib.LDA import LDA_Model
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Cargar rutas
directories = sorted([os.path.join("out", f) for f in os.listdir("out") if f.endswith(".csv")])
print("[INFO] Directorios detectados",directories)
def models():
    for file in directories:
        start_time =time.time()
        print(f"\n[INFO] Procesando dataset: {file}")

        df = pd.read_csv(file)
        scaler = StandardScaler()
        X = df.drop(columns=["labels"])
        X_scaled = scaler.fit_transform(X)
        y = df["labels"]
        pca=PCA(n_components=0.9)
        X_pca=pca.fit_transform(X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, train_size=0.8, stratify=y, random_state=42
        )

        file_name = file[3:-4]
        
        os.makedirs(os.path.join("logs","LDA"),exist_ok=True)
        os.makedirs(os.path.join("plots/confussion","LDA"),exist_ok=True)
        os.makedirs(os.path.join("plots/learning_curve","LDA"),exist_ok=True)
        os.makedirs(os.path.join("plots/ROC","LDA"),exist_ok=True)
        # ======== LDA ========
        lda = LDA_Model(solver="lsqr")
        lda.train(X_train, y_train)
        lda.evaluate(X_test, y_test, file_name)
        elapsed=time.time()-start_time 
        metrics = lda.metrics
        metrics["train_time"]=elapsed 
        print(f"[RESULT] [LDA] -> {metrics['accuracy']:.3f} -> {metrics}")
        save_logs_ml(metrics, f"logs/LDA/{file_name}.txt")
        print(f"[INFO] log almacenado en LDA{file_name}")
        plot_learning_curve(lda.model, X_pca, y, model_name="LDA" + file_name,path=f"plots/learning_curve")

models()
