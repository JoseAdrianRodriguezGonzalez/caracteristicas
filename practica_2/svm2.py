from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from lib.SVM import SVM_Lineal, SVM_kernel
from utils.plots import plot_learning_curve
from utils.metrics import evaluate_model
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
directories_train = sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "train" in f])[2:4]
directories_val=sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "val" in f])[2:4]

def models():
    for train,validation in zip(directories_train,directories_val):
        print(f"\n[INFO] Procesando dataset: {train}-{validation}")
        df_train = pd.read_csv(train)
        df_val = pd.read_csv(validation)
        df=pd.concat([df_train,df_val],ignore_index=True)
        scaler=StandardScaler()
        X = df.drop(columns=["label"])
        X_scaled=scaler.fit_transform(X)
        y = df["label"]
        lda=LDA()
        X_lda=lda.fit_transform(X_scaled,y)
        explained_ratios = lda.explained_variance_ratio_
        cumulative = np.cumsum(explained_ratios)
        n_comp = np.searchsorted(cumulative, 0.90) + 1
        print(f"[INFO] Componentes necesarios para 90% de 'varianza discriminante': {n_comp}")
        lda=LDA(n_components=n_comp)
        X_lda=lda.fit_transform(X_scaled,y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_lda, y, train_size=0.8, stratify=y, random_state=42
        )
        # ======== SVM MODELOS ========
        
        file_name=train[9:-4]+"lda"+str(n_comp)
        svm_lineal = SVM_Lineal()
        svm_lineal.train(X_train, y_train)
        svm_lineal.evaluate(X_test, y_test,file_name)
        metrics=svm_lineal.metrics
        print(f"[RESULT] [SVM_lineal] -> {metrics['accuracy']:.3f} -> {metrics}")
        save_logs_ml(metrics,f"logs/{svm_lineal.name+file_name}.txt")
        print(f"[INFO] log almacenado en {svm_lineal.name+file_name}")
        plot_learning_curve(svm_lineal.model, X_lda, y, model_name="svm_lineal"+file_name)

        svm_configs = [
            {"model": "SVM Quadratic", "kernel": "poly", "degree": 2,"C":1.0,"gamma":"scale","cache_size":1000, "tol":1e-3},
            {"model": "SVM RBF", "kernel": "rbf","C":1.0,"gamma":"scale","cache_size":1000, "tol":1e-3},
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
            plot_learning_curve(svm_kernel.model, X_lda, y, model_name=config['model']+file_name)
models()
