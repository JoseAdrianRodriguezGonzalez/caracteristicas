from lib.Knn import Knn
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.decomposition import PCA
directories = sorted([os.path.join("parte_1/data", f) for f in os.listdir("parte_1/data") if f.endswith(".csv")],reverse=True)
print("[INFO] Directorios detectados",directories)
def process_dataset(file):
    start_time =time.time()
    print(f"\n[INFO] Procesando dataset: {file}")
    df = pd.read_csv(file)
    scaler=StandardScaler()
    nan_cols = df.isna().sum()
    if nan_cols[nan_cols > 0].any():
        print(f"[WARN] Columnas con NaN en {file}:\n{nan_cols[nan_cols > 0]}")        
    X = df.drop(columns=["label"])
    X_scaled=scaler.fit_transform(X)
    pca=PCA(n_components=.90)
    #X_pca=pca.fit_transform(X_scaled)
    y = df["label"]
    train_size=0.7 if len(df)>500000 else 0.8
    X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=train_size, stratify=y, random_state=42
    )
    knn_configs = [
            {"model": "KNN_Fine", "n_neighbors": 3},
            {"model": "KNN_Medium", "n_neighbors": 5},
            {"model": "KNN_Coarse", "n_neighbors": 10},
        ]
    file_name=file[13:-4]

    for config in knn_configs:
        try:
            print(f"[INFO] Entrenando {config['model']} en {file_name}")
            knn_model = Knn(config)
            t0=time.time()
            knn_model.train(X_train, y_train)
            knn_model.evaluate(X_test,y_test,file_name)
            elapsed=time.time()-t0
            metrics = knn_model.metrics
            metrics["train_time"]=elapsed
            print(f"[RESULT] {config['model']} -> Acc:{metrics['accuracy']:.3f}| tiempo: {elapsed:.2f}s -> {metrics}")
            save_logs_ml(metrics,f"logs/{knn_model.name+file_name}_pca.txt")
            print(f"[INFO] log almacenado en {knn_model.name+file_name}")
            plot_learning_curve(knn_model.model, X_scaled, y, model_name=config['model']+file_name+"pca")
        except Exception as e:
            print(f"[ERROR] Falló {config['model']} en {file_name}: {e}")
    print(f"[INFO] Dataset {file_name} completado en {time.time() - start_time:.2f}s")
def models_parrallel():
    max_workers=os.cpu_count()//2
    print(f"[INFO] usando {max_workers} procesos en paralelo")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_dataset, directories), total=len(directories), desc="Procesando datasets"))
for directory in directories:
    process_dataset(directory)
