from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from lib.SVM import SVM_Lineal, SVM_kernel
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
# ======== LISTA DE DATASETS ========
directories = sorted([os.path.join("out", f) for f in os.listdir("out") if f.endswith(".csv")])
print("[INFO] Directorios detectados", directories)
def process_dataset(file):
    start_time = time.time()
    print(f"\n[INFO] Procesando dataset: {file}")
    
    df = pd.read_csv(file)
    scaler = StandardScaler()
    X = df.drop(columns=["labels"])
    y = df["labels"]
    X_scaled = scaler.fit_transform(X)
    pca=PCA(n_components=.90)

    X_pca=pca.fit_transform(X_scaled)
    # Tamaño de train dinámico
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, train_size=0.8, stratify=y, random_state=42
    )

    file_name = file[3:-4]

    # ======== SVM Lineal ========
    
    os.makedirs(os.path.join("logs","SVM_Lineal"),exist_ok=True)
    os.makedirs(os.path.join("plots/confussion","SVM_Lineal"),exist_ok=True)
    os.makedirs(os.path.join("plots/learning_curve","SVM_Lineal"),exist_ok=True)
    os.makedirs(os.path.join("plots/ROC","SVM_Lineal"),exist_ok=True)
    try:
        print(f"[INFO] Entrenando SVM Lineal en {file_name}")
        t0 = time.time()
        svm_lineal = SVM_Lineal()
        svm_lineal.train(X_train, y_train)
        svm_lineal.evaluate(X_test, y_test, file_name)
        elapsed = time.time() - t0
        metrics = svm_lineal.metrics
        metrics["train_time"] = elapsed
        print(f"[RESULT] [SVM_Lineal] -> Acc:{metrics['accuracy']:.3f} | tiempo: {elapsed:.2f}s -> {metrics}")
        save_logs_ml(metrics, f"logs/{svm_lineal.name}/{file_name}.txt")
        plot_learning_curve(svm_lineal.model, X_pca, y, model_name="SVM_Lineal/"+file_name,path=f"plots/learning_curve/",train_sizes=np.linspace(0.1,1.0,5))
    except Exception as e:
        print(f"[ERROR] Falló SVM_Lineal en {file_name}: {e}")
    # ======== SVM Kernels ========
    svm_configs = [
        {"model": "SVM_Quadr", "kernel": "poly", "degree": 2,},
        {"model": "SVM_fifth", "kernel": "poly", "degree": 5,},
        {"model": "SVM_qub", "kernel": "poly", "degree": 3},
        {"model": "SVM_RBF", "kernel": "rbf","C":0.1},
    ]

    for config in svm_configs:
        os.makedirs(os.path.join("logs",config["model"]),exist_ok=True)
        os.makedirs(os.path.join("plots/confussion",config["model"]),exist_ok=True)
        os.makedirs(os.path.join("plots/learning_curve",config["model"]),exist_ok=True)
        os.makedirs(os.path.join("plots/ROC",config["model"]),exist_ok=True)
        
        try:
            print(f"[INFO] Entrenando {config['model']} en {file_name}")
            t0 = time.time()
            svm_kernel = SVM_kernel(config)
            svm_kernel.train(X_train, y_train)
            svm_kernel.evaluate(X_test, y_test, file_name)
            elapsed = time.time() - t0
            metrics = svm_kernel.metrics
            metrics["train_time"] = elapsed
            print(f"[RESULT] {config['model']} -> Acc:{metrics['accuracy']:.3f} | tiempo: {elapsed:.2f}s -> {metrics}")
            save_logs_ml(metrics, f"logs/{config['model']}/{file_name}.txt")
            plot_learning_curve(svm_kernel.model, X_pca, y, model_name=config['model']+file_name,path=f"plots/learning_curve/")
        except Exception as e:
            print(f"[ERROR] Falló {config['model']} en {file_name}: {e}")

    print(f"[INFO] Dataset {file_name} completado en {time.time() - start_time:.2f}s")


for directory in directories:
    process_dataset(directory)
