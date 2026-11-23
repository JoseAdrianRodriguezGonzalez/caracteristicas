# RF.py
from lib.RandomForest  import RandomForest
from utils.plots import plot_learning_curve
from utils.metrics import evaluate_model
from utils.save_metrics import save_logs_ml

import pandas as pd
import numpy as np
import os
import time
import matplotlib 
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
# Obtener todos los archivos CSV
directories = sorted([os.path.join("out", f) for f in os.listdir("out") if f.endswith(".csv")])


def run_random_forest():

    for file in directories:
        print(f"\n[INFO] Procesando dataset: {file}")
        start = time.time()

        df = pd.read_csv(file)

        X = df.drop(columns=["labels"])
        y = df["labels"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)  # conserva 95% varianza
        X_pca = pca.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, train_size=0.8, stratify=y, random_state=42
        )

        file_name = file[3:-4]

        # === Directorios ===
        model_dir = "RandomForest"
        os.makedirs(f"logs/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/confussion/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/learning_curve/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/ROC/{model_dir}", exist_ok=True)

        # === Hiperparámetros del RF ===
        params = {
            "model": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features":"sqrt",
            "bootstrap": True,
        }

        # === Entrenar RF ===
        rf = RandomForest(**params)

        t0 = time.time()
        rf.train(X_train, y_train)
        train_time = time.time() - t0

        # === Evaluar ===
        rf.evaluate(X_test, y_test, file_name)

        metrics = rf.metrics
        metrics["train_time"] = train_time

        save_logs_ml(metrics, f"logs/{rf.name}/{file_name}.txt")

        print(metrics)

        # === Learning Curve ===
        plot_learning_curve(
            rf.model,
            X_pca,
            y,
            model_name=rf.name + file_name,
            path="plots/learning_curve/"
        )

        print("[INFO] Evaluación completada para este dataset.")


run_random_forest()

