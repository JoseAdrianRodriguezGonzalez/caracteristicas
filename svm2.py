from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from lib.SVM import SVM_Lineal, SVM_kernel
from utils.plots import plot_learning_curve
from utils.metrics import evaluate_model
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
directories = [os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv")]

def models():
    for file in directories:
        print(f"\n[INFO] Procesando dataset: {file}")
        df = pd.read_csv(file)
        scaler=StandardScaler()
        X = df.drop(columns=["label"])
        X_scaled=scaler.fit_transform(X)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=0.8, stratify=y, random_state=42
        )
        # ======== SVM MODELOS ========
        svm_lineal = SVM_Lineal()
        svm_lineal.train(X_train, y_train)
        file_name=file[9:-4]
        svm_lineal.evaluate(X_test, y_test,file_name)
        metrics=svm_lineal.metrics
        print(f"[RESULT] [SVM_lineal] -> {metrics['accuracy']:.3f} -> {metrics}")
        save_logs_ml(metrics,f"logs/{svm_lineal.name+file_name}.txt")
        print(f"[INFO] log almacenado en {svm_lineal.name+file_name}")
        plot_learning_curve(svm_lineal.model, X, y, model_name="svm_lineal"+file_name)
        svm_configs = [
            {"model": "SVM Quadratic", "kernel": "poly", "degree": 2},
            {"model": "SVM RBF", "kernel": "rbf"},
        ]
        for config in svm_configs:
            svm_kernel = SVM_kernel(config)
            print(f"[INFO] Entrenando {config['model']}")
            svm_kernel.train(X_train, y_train)
            svm_kernel.evaluate(X_test, y_test,file_name)
            metrics=svm_kernel.metrics
            print(f"[RESULT] {config['model']} -> {metrics['accuracy']:.3f} -> {metrics}")
            save_logs_ml(metrics,f"logs/{config['model']+file_name}.txt")
            print(f"[INFO] log almacenado en {config['model']+file_name}")
            plot_learning_curve(svm_kernel.model, X_scaled, y, model_name=config['model']+file_name)
models()
