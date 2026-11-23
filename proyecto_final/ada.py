from lib.Adaboost import AdaBoost
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
directories = sorted([os.path.join("out", f) for f in os.listdir("out") if f.endswith(".csv")])
def run_adaboost():
    for file in directories:
        print(f"\n[INFO] Procesando dataset: {file}")
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

        model_dir = "AdaBoost"
        os.makedirs(f"logs/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/confussion/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/learning_curve/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/ROC/{model_dir}", exist_ok=True)

        # === Hiperparámetros por defecto ===
        params = {
            "model": "AdaBoost",
            "n_estimators": 300,
            "learning_rate": 0.2
        }

        model = AdaBoost(**params)

        t0 = time.time()
        model.train(X_train, y_train)
        elapsed = time.time() - t0

        model.evaluate(X_test, y_test, file_name)

        metrics = model.metrics
        metrics["train_time"] = elapsed
        save_logs_ml(metrics, f"logs/{model.name}/{file_name}.txt")

        print(metrics)

        plot_learning_curve(
            model.model, X_pca, y,
            model_name=model.name + file_name,
            path="plots/learning_curve/"
        )

        print("[INFO] Evaluación completada.")


run_adaboost()
