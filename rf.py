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
from lib.RandomForest import RandomForest
directories = sorted([os.path.join("parte_1/data", f) for f in os.listdir("parte_1/data") if f.endswith(".csv")],reverse=True)
print("[INFO] Directorios detectados",directories)

def process_dataset(file):
    start_time = time.time()
    print(f"\n[INFO] Procesando dataset: {file}")
    df = pd.read_csv(file)
    scaler = StandardScaler()
    nan_cols = df.isna().sum()
    if nan_cols[nan_cols > 0].any():
        print(f"[WARN] Columnas con NaN en {file}:\n{nan_cols[nan_cols > 0]}")
    
    X = df.drop(columns=["label"])
    X_scaled = scaler.fit_transform(X)
    y = df["label"]

    train_size = 0.7 if len(df) > 500000 else 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=train_size, stratify=y, random_state=42
    )

    # Configuraciones de Random Forest
    rf_configs = [
        {"model": "RF_Small", "n_estimators": 50, "max_depth": 10},
        {"model": "RF_Medium", "n_estimators": 100, "max_depth": 20},
        {"model": "RF_Large", "n_estimators": 200, "max_depth": None},
    ]

    file_name = file[13:-4]

    for config in rf_configs:
        try:
            print(f"[INFO] Entrenando {config['model']} en {file_name}")
            rf_model = RandomForest(config)
            t0 = time.time()
            rf_model.train(X_train, y_train)
            rf_model.evaluate(X_test, y_test, file_name)
            elapsed = time.time() - t0
            metrics = rf_model.metrics
            metrics["train_time"] = elapsed
            print(f"[RESULT] {config['model']} -> Acc:{metrics['accuracy']:.3f} | tiempo: {elapsed:.2f}s -> {metrics}")
            save_logs_ml(metrics, f"logs/{rf_model.name + file_name}.txt")
            print(f"[INFO] log almacenado en {rf_model.name + file_name}")
            plot_learning_curve(rf_model.model, X_scaled, y, model_name=config['model'] + file_name)
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
