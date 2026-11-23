from lib.DecisionTree import DecisionTree
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


def run_decision_tree():

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
        model_dir = "DecisionTree"
        os.makedirs(f"logs/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/confussion/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/learning_curve/{model_dir}", exist_ok=True)
        os.makedirs(f"plots/ROC/{model_dir}", exist_ok=True)

        # === Hiperparámetros ===
        params = {
            "model": "DecisionTree",
            "criterion": "gini",
            "max_depth": 50,
            "min_samples_split": 10,
            "min_samples_leaf": 20,
            "max_features":"sqrt"
        }
        # === Entrenar ===
        dt = DecisionTree(**params)

        t0 = time.time()
        dt.train(X_train, y_train)
        train_time = time.time() - t0

        # === Evaluar ===
        dt.evaluate(X_test, y_test, file_name)

        metrics = dt.metrics
        metrics["train_time"] = train_time

        save_logs_ml(metrics, f"logs/{dt.name}/{file_name}.txt")

        print(metrics)

        # === Learning Curve ===
        plot_learning_curve(
            dt.model,
            X_pca,
            y,
            model_name=dt.name + file_name,
            path="plots/learning_curve/"
        )

        print("[INFO] Evaluación completada para este dataset.")



run_decision_tree()
