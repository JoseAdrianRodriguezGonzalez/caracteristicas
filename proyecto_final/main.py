from utils.sort import labeling_2, list_files,labeling, count_samples_txt,count_samples_txt_2
from utils.create_csv import create_csv
import os
from src.image_preprocessing import read_images, read_images_2
from src.features import read_from_txt, read_image_from_txt_features
path_base="data/experiment-i"
def parte_1(path:str):
    #pacientes=list_files(path)
    #left_=["3","6","7","14"]
    #right_=["2","4","5","13"]
    #suppine_=["1","8","9","10","11","12","15","16","17"]
    #labeling(path,pacientes,left_,right_,suppine_)
    izquierda=count_samples_txt(os.path.join(path,"left"))
    derecha=count_samples_txt(os.path.join(path,"right"))
    suppine=count_samples_txt(os.path.join(path,"suppine"))
    print(izquierda,derecha,suppine)
   # read_images(path)  # augmentation
    features,labels=read_from_txt(path)
    create_csv(features,labels,"out/dataset_experimento_i.csv")
def parte_2(path:str):
    pacientes=list_files(path)
    #left_=["D1","D2","D3","E3","E4","E6"]
    #right_=["C1","C2","C3","E1","E2","E5"]
    #suppine_=["B1","B2","B3","B4","B5","B6","B7","B8","B9","10","F1","F2","F3","F4","F5","F6","F7"]
    #labeling_2(path,pacientes,left_,right_,suppine_)
    directorio=["Air_Mat","Sponge_Mat"]
    for carpeta in directorio:
        izquierda=count_samples_txt_2(os.path.join(path,carpeta,"left"))
        derecha=count_samples_txt_2(os.path.join(path,carpeta,"right"))
        suppine=count_samples_txt_2(os.path.join(path,carpeta,"suppine"))
       # read_images_2(os.path.join(path,carpeta)) 
        print(izquierda,derecha,suppine)
        features,labels=read_image_from_txt_features(os.path.join(path,carpeta))  
        create_csv(features,labels,f"out/dataset_{carpeta}.csv")

#parte_1("data/experiment-i")    
parte_2("data/experiment-ii")
import pandas as pd 
df=pd.read_csv("out/dataset_experimento_i.csv")
print(df.isna().sum())
