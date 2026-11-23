import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import learning_curve,StratifiedKFold 
from sklearn.metrics import roc_curve,auc
def plot_multiclass_roc(y_true, y_prob, classes, model_name="Model", path="results/roc"):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(7,6))
    for i, cls in enumerate(classes):
        fpr,tpr,_ = roc_curve(y_true == cls, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0,1], [0,1], '--', label="Azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curvas ROC Multiclase - {model_name}")
    plt.legend()
    plt.tight_layout()
    file = os.path.join(path, f"{model_name}_multiclass_roc.png")
    plt.savefig(file, dpi=200)
    plt.close()
    print(f"[INFO] Curva ROC multiclase guardada en {file}")
def plot_confusion_matrix(cm, classes, model_name="Model", path="results"):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.xlabel("Predicciones")
    plt.ylabel("Reales")
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plot_path = os.path.join(path, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[INFO] Matriz de confusión guardada en {plot_path}")
def plot_learning_curve(model,X,y,model_name="Model",path="Results",cv=5,train_sizes=None):
    if train_sizes is None:
        train_sizes=np.linspace(0.1,1.0,10)
    cv=StratifiedKFold(n_splits=cv,shuffle=True,random_state=42)
    train_sizes,train_scores,val_scores=learning_curve(
        model,X,y,
        cv=cv,
        train_sizes=train_sizes
        ,scoring="accuracy",
        n_jobs=-1)
    train_mean=np.mean(train_scores,axis=1)
    val_mean=np.mean(val_scores,axis=1)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Entrenamiento")
    plt.plot(train_sizes, val_mean, "s--", label="Validación")
    plt.title(f"Curva de Aprendizaje - {model_name}")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión (Accuracy)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    os.makedirs(path, exist_ok=True)
    plot_path = os.path.join(path, f"{model_name}_learning_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[INFO] Curva de aprendizaje guardada en {plot_path}")

