
from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from utils.plots import plot_learning_curve
from utils.metrics import evaluate_model
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

directories = sorted([os.path.join("parte_1/data", f) for f in os.listdir("parte_1/data") if f.endswith(".csv")],reverse=True)

def models():
    for file in directories:
        print(f"\n[INFO] Procesando dataset: {file}")
        df = pd.read_csv(file)

        X = df.drop(columns=["label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, stratify=y, random_state=42
        )

        # ======== GAUSSIAN NAIVE BAYES ========
        nb_gaussian = GaussianNaiveBayes()
        nb_gaussian.train(X_train, y_train)
        nb_gaussian.evaluate(X_test, y_test)
        print(nb_gaussian.metrics)
        plot_learning_curve(nb_gaussian.model, X, y, model_name="Gaussian")

        # ======== KERNEL NAIVE BAYES ========
        nb_kernel = KernelNaiveBayes(bandwidth=1.0)
        nb_kernel.fit(X_train.to_numpy(), y_train.to_numpy())
        nb_kernel.evaluate(X_test.to_numpy(), y_test.to_numpy())
        print(nb_kernel.metrics)
        # Para este modelo, podemos pasar el objeto completo:
        plot_learning_curve(nb_kernel, X.to_numpy(), y.to_numpy(), model_name="Kernel")

        print("[INFO] Evaluaciones completadas para este dataset.")

models()

