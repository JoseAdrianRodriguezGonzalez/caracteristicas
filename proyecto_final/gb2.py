

from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from utils.plots import plot_learning_curve
from utils.metrics import evaluate_model
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import time
directories = sorted([os.path.join("out", f) for f in os.listdir("out") if f.endswith(".csv")])
def models():
    for file in directories:
        print(f"\n[INFO] Procesando dataset: {file}")
        start_time =time.time()
        df = pd.read_csv(file)

        X = df.drop(columns=["labels"])
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X)
        y = df["labels"]
        pca = PCA(n_components=0.95)  # conserva 95% varianza
        X_pca = pca.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca , y, train_size=0.8, stratify=y, random_state=42
        )
        file_name=file[3:-4]
        os.makedirs(os.path.join("logs","NaiveBayesGaussian"),exist_ok=True)
        os.makedirs(os.path.join("plots/confussion","NaiveBayesGaussian"),exist_ok=True)
        os.makedirs(os.path.join("plots/learning_curve","NaiveBayesGaussian"),exist_ok=True)
        os.makedirs(os.path.join("plots/ROC","NaiveBayesGaussian"),exist_ok=True)
        # ======== GAUSSIAN NAIVE BAYES ========
        nb_gaussian = GaussianNaiveBayes()
        t0=time.time()
        nb_gaussian.train(X_train, y_train)
        nb_gaussian.evaluate(X_test, y_test,file_name)
        elapsed=time.time()-t0
        metrics=nb_gaussian.metrics
        metrics["train_time"] = elapsed
        save_logs_ml(metrics,f"logs/{nb_gaussian.name}/{file_name}.txt")
        print(nb_gaussian.metrics)
        plot_learning_curve(nb_gaussian.model, X_pca, y, model_name="NaiveBayesGaussian"+file_name,path="plots/learning_curve/")

        os.makedirs(os.path.join("logs","NaiveBayesKernel"),exist_ok=True)
        os.makedirs(os.path.join("plots/confussion","NaiveBayesKernel"),exist_ok=True)
        os.makedirs(os.path.join("plots/learning_curve","NaiveBayesKernel"),exist_ok=True)
        os.makedirs(os.path.join("plots/ROC","NaiveBayesKernel"),exist_ok=True)
        # ======== KERNEL NAIVE BAYES ========
        nb_kernel = KernelNaiveBayes(bandwidth=1.0)
        t0=time.time()
        nb_kernel.fit(X_train, y_train)
        nb_kernel.evaluate(X_test, y_test,file_name)
        elapsed=time.time()-t0
        print(nb_kernel.metrics)
        metrics=nb_kernel.metrics
        metrics["train_time"] = elapsed
        save_logs_ml(metrics,f"logs/{nb_kernel.name}/{file_name}.txt")
        # Para este modelo, podemos pasar el objeto completo:
        plot_learning_curve(nb_kernel, X_scaled, y, model_name="NaiveBayesKernel"+file_name,path="plots/learning_curve")

        print("[INFO] Evaluaciones completadas para este dataset.")

models()

