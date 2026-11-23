import os
import matplotlib.pyplot as plt
def save_logs(logs:dict,path):
    n = len(next(iter(logs.values())))
    with open(path,"w") as log:
        for i in range(n):
            line = " ".join(f"{k}: {v[i]}" for k, v in logs.items())
            log.write(line + "\n")
    print(f"log almacenado en {path}")

def save_logs_ml(logs, path):
    with open(path, "w") as f:
        for k, v in logs.items():
            f.write(f"{k}: {v}\n")
def plot_history(history,model="name",path="results"):
    keys=history.keys()
    epochs=range(1,len(history["train_acc"])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs,history["train_acc"],"o-",label="Entrenamiento")
    plt.plot(epochs,history["val_acc"],"s--",label="Validacion")
    plt.title(f"Accuacy - {model}")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True,linestyle="--",alpha=0.6)

    plt.subplot(1,2,2)
    plt.plot(epochs,history["train_loss"],"o-",label="Entrenamiento")
    plt.plot(epochs,history["val_loss"],"s--",label="Validacion")
    plt.title(f"Perdida - {model}")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida")
    plt.legend()
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.tight_layout()
    plot_path=os.path.join(path,f"{model}_history.png")
    plt.savefig(plot_path,dpi=200)
    plt.close()
    print(f"Grafica guardada en {plot_path}")

