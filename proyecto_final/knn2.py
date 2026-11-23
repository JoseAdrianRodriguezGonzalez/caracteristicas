from lib.Knn import Knn
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import time
import matplotlib 
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.decomposition import PCA
directories = sorted([os.path.join("out", f) for f in os.listdir("out") if f.endswith(".csv")])
print("[INFO] Directorios detectados",directories)
def process_dataset(file):
    start_time =time.time()
    print(f"\n[INFO] Procesando dataset: {file}")
    df = pd.read_csv(file)
    scaler=StandardScaler()
   # nan_cols = df.isna().sum()
    #if nan_cols[nan_cols > 0].any():
    #    print(f"[WARN] Columnas con NaN en {file}:\n{nan_cols[nan_cols > 0]}")        
    X = df.drop(columns=["labels"])
    X_scaled=scaler.fit_transform(X)
    pca=PCA(n_components=.90)
    X_pca=pca.fit_transform(X_scaled)
    y = df["labels"]
    X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, train_size=0.7, stratify=y, random_state=42
    )
    knn_configs = [
             {"model": "KNN_Fine", "n_neighbors": 3},
             {"model": "KNN_Medium", "n_neighbors": 5},
             {"model": "KNN_Coarse", "n_neighbors": 20},
             {"model": "KNN_Weighted", "n_neighbors": 10, "weights": "distance"},   # ponderado
             {"model": "KNN_Minkowski", "n_neighbors": 10, "metric": "minkowski"},  # métrica Minkowski (p=2 es euclídea)
            {"model": "KNN_Cosine", "n_neighbors": 10, "metric": "cosine"}
        ]
    file_name=file[3:-4]
    
    for config in knn_configs:
        os.makedirs(os.path.join("logs",config["model"]),exist_ok=True)
        os.makedirs(os.path.join("plots/confussion",config["model"]),exist_ok=True)
        os.makedirs(os.path.join("plots/learning_curve",config["model"]),exist_ok=True)
        os.makedirs(os.path.join("plots/ROC",config["model"]),exist_ok=True)
        try:
            print(f"[INFO] Entrenando {config['model']} en {file_name}")
            knn_model = Knn(config)
            t0=time.time()
            knn_model.train(X_train, y_train)
            knn_model.evaluate(X_test,y_test,f"{file_name}")
            elapsed=time.time()-t0
            metrics = knn_model.metrics
            metrics["train_time"]=elapsed
            print(f"[RESULT] {config['model']} -> Acc:{metrics['accuracy']:.3f}| tiempo: {elapsed:.2f}s -> {metrics}")
            save_logs_ml(metrics,f"logs/{knn_model.name}/{file_name}.txt")
            print(f"[INFO] log almacenado en logs/{knn_model.name}/{file_name}.txt")
            plot_learning_curve(knn_model.model, X_pca, y, model_name=config['model']+file_name,path=f"plots/learning_curve/")
        except Exception as e:
            print(f"[ERROR] Falló {config['model']} en {file_name}: {e}")
    print(f"[INFO] Dataset {file_name} completado en {time.time() - start_time:.2f}s")
for directory in directories:
    process_dataset(directory)
