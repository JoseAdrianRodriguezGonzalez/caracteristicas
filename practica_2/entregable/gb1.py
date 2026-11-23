
from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from utils.plots import plot_learning_curve
from utils.metrics import evaluate_model
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
directories = sorted([os.path.join("parte_1/data", f) for f in os.listdir("parte_1/data") if f.endswith(".csv")],reverse=True)

def models():
    for file in directories:
        print(f"\n[INFO] Procesando dataset: {file}")
        df = pd.read_csv(file)

        X = df.drop(columns=["label"])
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X)
        y = df["label"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=0.8, stratify=y, random_state=42
        )
        file_name=file[13:-4]
        # ======== GAUSSIAN NAIVE BAYES ========
        nb_gaussian = GaussianNaiveBayes()
        nb_gaussian.train(X_train, y_train)
        nb_gaussian.evaluate(X_test, y_test,file_name)
        metrics=nb_gaussian.metrics
        save_logs_ml(metrics,f"logs/Gaussian{file_name}.txt")
        print(nb_gaussian.metrics)
        plot_learning_curve(nb_gaussian.model, X_scaled, y, model_name="Gaussian"+file_name)

        # ======== KERNEL NAIVE BAYES ========
        nb_kernel = KernelNaiveBayes(bandwidth=1.0)
        nb_kernel.fit(X_train, y_train)
        nb_kernel.evaluate(X_test, y_test,file_name)
        print(nb_kernel.metrics)
        metrics=nb_kernel.metrics
        save_logs_ml(metrics,f"logs/kernel_{file_name}.txt")
        # Para este modelo, podemos pasar el objeto completo:
        plot_learning_curve(nb_kernel, X_scaled, y, model_name="Kernel"+file_name)

        print("[INFO] Evaluaciones completadas para este dataset.")

models()

