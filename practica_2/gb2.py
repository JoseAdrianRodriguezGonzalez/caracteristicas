
from lib.Bayes import KernelNaiveBayes, GaussianNaiveBayes
from utils.plots import plot_learning_curve
from utils.save_metrics import save_logs_ml
from utils.metrics import evaluate_model
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
directories_train = sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "train" in f])[2:4]
directories_val=sorted([os.path.join("features/", f) for f in os.listdir("features") if f.endswith(".csv") and "val" in f])[2:4]
def models():
    for train,validation in zip(directories_train,directories_val):
        print(f"\n[INFO] Procesando dataset: {train}-{validation}")
        df_train=pd.read_csv(train)
        print(df_train.shape)
        df_val=pd.read_csv(validation)
        print(df_val.shape)
        df=pd.concat([df_train,df_val],ignore_index=True)
        print(df.shape)
        scaler=StandardScaler()
        nan_cols=df.isna().sum()
        print(nan_cols[nan_cols>0])
        
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
        file_name=train[9:-4]+"lda_"+str(n_comp)
        print(file_name)
        # ======== GAUSSIAN NAIVE BAYES ========
        
        nb_gaussian = GaussianNaiveBayes()
        nb_gaussian.train(X_train, y_train)
        nb_gaussian.evaluate(X_test, y_test,file_name)
        print(nb_gaussian.metrics)
        plot_learning_curve(nb_gaussian.model, X_lda, y, model_name="Gaussian"+file_name)
        metrics=nb_gaussian.metrics
        save_logs_ml(metrics,f"logs/cnn/{nb_gaussian.name+file_name}.txt")
        print(f"[INFO] log almacenado en {nb_gaussian.name+file_name}")
        
        # ======== KERNEL NAIVE BAYES ========
        
        nb_kernel = KernelNaiveBayes(bandwidth=1.0)
        nb_kernel.fit(X_train, y_train)
        nb_kernel.evaluate(X_test, y_test,file_name)
        print(nb_kernel.metrics)
        metrics=nb_kernel.metrics
        save_logs_ml(metrics,f"logs/cnn/{nb_kernel.name+file_name}.txt")
        print(f"[INFO] log almacenado en {nb_kernel.name+file_name}")
        # Para este modelo, podemos pasar el objeto completo:
        plot_learning_curve(nb_kernel, X_lda, y, model_name="Kernel"+file_name)
        
        print("[INFO] Evaluaciones completadas para este dataset.")

models()

