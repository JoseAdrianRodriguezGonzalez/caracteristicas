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
directories = sorted(
    [os.path.join("parte_1/data", f) for f in os.listdir("parte_1/data") if f.endswith(".csv")],
    reverse=True
)
print("[INFO] Directorios detectados", directories)


def process_dataset(file):
    start_time = time.time()
    print(f"\n[INFO] Procesando dataset: {file}")
    
    df = pd.read_csv(file)
    scaler = StandardScaler()

    nan_cols = df.isna().sum()
    if nan_cols[nan_cols > 0].any():
        print(f"[WARN] Columnas con NaN en {file}:\n{nan_cols[nan_cols > 0]}")

    X = df.drop(columns=["label"])
    y = df["label"]
    X_scaled = scaler.fit_transform(X)
#    X_pca=PCA(n_components=.90)

    # Tamaño de train dinámico
    train_size = 0.7 if len(df) > 500000 else 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=train_size, stratify=y, random_state=42
    )

    file_name = file[13:-4]

    # ======== SVM Lineal ========
    """
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
        save_logs_ml(metrics, f"logs/{svm_lineal.name+file_name}_pca.txt")
        plot_learning_curve(svm_lineal.model, X_scaled, y, model_name="svm_lineal_"+file_name)
    except Exception as e:
        print(f"[ERROR] Falló SVM_Lineal en {file_name}: {e}")
"""
    # ======== SVM Kernels ========
    svm_configs = [
        {"model": "SVM fifth", "kernel": "poly", "degree": 5,},
        #{"model": "SVM qub", "kernel": "poly", "degree": 3},
   #     {"model": "SVM RBF", "kernel": "rbf","C":0.1},
    ]

    for config in svm_configs:
        try:
            print(f"[INFO] Entrenando {config['model']} en {file_name}")
            t0 = time.time()
            svm_kernel = SVM_kernel(config)
            #svm_kernel.train(X_train, y_train)
            #svm_kernel.evaluate(X_test, y_test, file_name)
            elapsed = time.time() - t0
            #metrics = svm_kernel.metrics
            #metrics["train_time"] = elapsed
            #print(f"[RESULT] {config['model']} -> Acc:{metrics['accuracy']:.3f} | tiempo: {elapsed:.2f}s -> {metrics}")
            #save_logs_ml(metrics, f"logs/{config['model']+file_name}.txt")
            plot_learning_curve(svm_kernel.model, X_scaled, y, model_name=config['model']+file_name)
        except Exception as e:
            print(f"[ERROR] Falló {config['model']} en {file_name}: {e}")

    print(f"[INFO] Dataset {file_name} completado en {time.time() - start_time:.2f}s")


def models_parallel():
    max_workers = os.cpu_count() // 2
    print(f"[INFO] usando {max_workers} procesos en paralelo")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_dataset, directories),
                  total=len(directories),
                  desc="Procesando datasets"))

for directory in directories:
    process_dataset(directory)
