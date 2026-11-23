import torch
import torch.nn as nn 
import torch.optim as optim 
from parte_2.aumentation import apply_filter
from parte_2.load_dataset import load_dataset
from parte_2.train import Trainer
from parte_2.models.vgg import build_vgg,build_vgg_extractor,build_vgg_extractor_fc
from parte_2.models.efficient import build_efficientnet,build_efficientnet_extractor,build_efficientnet_extractor_fc
from utils.save_metrics import save_logs,plot_history
import numpy as np
import pandas as pd
import os
def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    X = np.concatenate(features)
    y = np.concatenate(labels)
    return X, y

def aumentacion(input_dir,output_dir,quantity):
    error_code=apply_filter(input_dir,output_dir,quantity)
    if not error_code:
        raise ValueError("No se procesaron las imagenes de forma correcta, codigo de error",1)
def load_dataset_images(
    directory:str="data",
    batch_size:int=8,
    image_size:tuple[int,int]=(224,224),
    num_workers:int=4,
    mean:list[float]=[0.485,0.456,0.406],
    std:list[float]=[0.229,0.224,0.225],
    split_ratio:float=0.8,
    
):

    return load_dataset(directory,batch_size,image_size,num_workers,mean,std,split_ratio)
def save_features_to_csv(X,y, filename_prefix,output_dir="features"):
    os.makedirs(output_dir,exist_ok=True)
    df=pd.DataFrame(X)
    df["label"]=y
    csv_path=os.path.join(output_dir,f"{filename_prefix}.csv")
    df.to_csv(csv_path,index=False)
    print(f"[OK] Features guardadas en {csv_path}")

MODEL_CONFIG = {
    # EfficientNet
#    "EfficientNet_B0": {"batch_size": 32, "freeze_until": 5},
 #   "EfficientNet_B1": {"batch_size": 32, "freeze_until": 5},
 #   "EfficientNet_B2": {"batch_size": 32, "freeze_until": 6},
 #   "EfficientNet_B3": {"batch_size": 32, "freeze_until": 6},
 #   "EfficientNet_B4": {"batch_size": 8, "freeze_until": 7},
 #   "EfficientNet_B5": {"batch_size": 16,  "freeze_until": 7},
    "EfficientNet_B6": {"batch_size": 8,  "freeze_until": 7},
    "EfficientNet_B7": {"batch_size": 8,  "freeze_until": 7},

    # VGG
    "VGG11": {"batch_size": 32, "freeze_until": 10},
    "VGG13": {"batch_size": 32, "freeze_until": 16}, 
    "VGG16": {"batch_size": 32, "freeze_until": 24},  
    "VGG19": {"batch_size": 32, "freeze_until": 28},
}
def check_dataset(dataloader):
    class_counts = {}
    total_images = 0

    for i, (imgs, lbls) in enumerate(dataloader):
        batch_size = imgs.size(0)
        total_images += batch_size

        for c in lbls.numpy():
            class_counts[c] = class_counts.get(c, 0) + 1

    print(f"[INFO] Total de imágenes en el DataLoader: {total_images}")
    print(f"[INFO] Conteo por clase: {class_counts}")
    return class_counts
def main(model_name:str,freeze_until:int):
    
    #----------------Section 2 ------------------
    #---------Aumentation section--------------
  #  input_dir="parte_2/data/Pedestrian"
  #  output_dir="parte_2/data/Pedestrian"
  #  aumentacion(input_dir,output_dir,600)
    #--------Convolutional section-------------

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando {device}")
    #data:w
    
    train_l ,validation_l,classes=load_dataset_images(directory="parte_2/data",batch_size=MODEL_CONFIG[model_name]['batch_size'])
    print("[CHECK] Conjunto de entrenamiento")
    check_dataset(train_l)

    print("[CHECK] Conjunto de validación")
    check_dataset(validation_l)
    
    #extractor 
    #val=6
    print(f"EL modelo {model_name} con congelamiento en {freeze_until}")
    if "EfficientNet" in model_name:
        model=build_efficientnet(model_name,len(classes),freeze_until)
    elif "VGG" in model_name:
        model=build_vgg(model_name.lower(),len(classes),freeze_until)
    else:
        raise ValueError(f"Modelo {model_name} no reconocido")
    #model=build_vgg("vgg13",len(classes),28)
    #optimizer and criterion
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-4,weight_decay=1e-4)
    trainer=Trainer(model,train_l,validation_l,device,criterion,optimizer)   
    trainer.fit()
    save_logs(trainer.history,f"logs/{model_name}.txt")
    plot_history(trainer.history,model_name,"plots/")


    if "EfficientNet" in model_name:
        extractor=build_efficientnet_extractor(model_name,freeze_until).to(device)
    else:
        extractor=build_vgg_extractor(model_name.lower(),freeze_until).to(device)
    #extreaer features 
    print("[INFO] Extrayendo features del conjunto de entrenamiento...")
    X_train, y_train = extract_features(extractor, train_l, device)
    print("[INFO] Extrayendo features del conjunto de validación...")
    X_val, y_val = extract_features(extractor, validation_l, device)

    # Guardar en CSV
    
    save_features_to_csv(X_train, y_train, f"{model_name}_train_features")
    save_features_to_csv(X_val, y_val, f"{model_name}_val_features")


    #Fully coneccted
    #model
    if "EfficientNet" in model_name:
        extractor=build_efficientnet_extractor_fc(model_name,freeze_until).to(device)
    else:
        extractor=build_vgg_extractor_fc(model_name.lower(),freeze_until).to(device)
    print("[INFO] Extrayendo la seccion del fc del conjunto de entrenamiento...")
    X_train, y_train = extract_features(extractor, train_l, device)
    print("[INFO] Extrayendo la seccion del fc del conjunto de validación...")
    X_val, y_val = extract_features(extractor, validation_l, device)

    # Guardar en CSV
    
    save_features_to_csv(X_train, y_train, f"{model_name}_train_fully_connected")
    save_features_to_csv(X_val, y_val,  f"{model_name}_val_fully_connected")
if __name__ =="__main__":
    for models,values in MODEL_CONFIG.items():
        main(models,values['freeze_until'])
