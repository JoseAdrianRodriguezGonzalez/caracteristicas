import numpy as np 
import os 
from utils.sort import count_samples_txt,count_samples_txt_2
import random as rnd 
from lib.augmentation import apply_random_augmentation
def read_images(root_path:str):
    labels=["left","right",'suppine']
    izquierda=count_samples_txt(os.path.join(root_path,"left"))
    derecha=count_samples_txt(os.path.join(root_path,"right"))
    suppine=count_samples_txt(os.path.join(root_path,"suppine"))
    print(f"{suppine}->{izquierda}->{derecha}")
    left_augmentation(suppine,izquierda,os.path.join(root_path,labels[0]))
    right_augmentation(suppine,derecha,os.path.join(root_path,labels[1]))
    izquierda=count_samples_txt(os.path.join(root_path,"left"))
    derecha=count_samples_txt(os.path.join(root_path,"right"))
    print(f"{suppine}->{izquierda}->{derecha}")
def read_images_2(root_path:str):
    labels=["left","right",'suppine']
    izquierda=count_samples_txt_2(os.path.join(root_path,"left"))
    derecha=count_samples_txt_2(os.path.join(root_path,"right"))
    suppine=count_samples_txt_2(os.path.join(root_path,"suppine"))
    print(f"Antes: {suppine}->{izquierda}->{derecha}")
    left_augmentation_2(420,izquierda,os.path.join(root_path,labels[0]))
    right_augmentation_2(420,derecha,os.path.join(root_path,labels[1]))
    suppine_augmentation_2(420,suppine,os.path.join(root_path,labels[2]))
    izquierda=count_samples_txt(os.path.join(root_path,"left"))
    derecha=count_samples_txt(os.path.join(root_path,"right"))
    print(f"{suppine}->{izquierda}->{derecha}")
def load_image_from_txt(path):
    with open(path,"r") as f:
        lines=f.readlines()
    matrix=[list(map(float,line.strip().split())) for line in lines]
    return np.array(matrix) 
def left_augmentation_2(suppine_quantity_:int,left:int,root_path:str):
    n=suppine_quantity_-left
    print(n)
    files=[file for file in sorted(os.listdir(root_path))]
    for i in range(n):
    #    print(n)
        choosen_file=rnd.choice(files)
        print(choosen_file)
        matriz=load_image_from_txt(os.path.join(root_path,choosen_file))
        matriz_augmented=apply_random_augmentation(matriz)
        newname=choosen_file.replace(".txt",f"_aug{i}_l.txt")
        with open(os.path.join(root_path,newname),"w") as f:
            for row in matriz_augmented:
                f.write(" ".join(str(x) for x in row) + "\n")
        #array_augmented=generate_augmented_image(chosen)
        #line= " ".join(str(x) for x in array_augmented)+'\n'
        #with open(os.path.join(root_path,choosen_file),"a") as f:
        #    f.write(line)
def suppine_augmentation_2(suppine_quantity_:int,right:int,root_path:str):
    n=suppine_quantity_-right
    print(n)
    files=[file for file in sorted(os.listdir(root_path))]
    for i in range(n):
    #    print(n)
        choosen_file=rnd.choice(files)
        print(choosen_file)
        matriz=load_image_from_txt(os.path.join(root_path,choosen_file))
        matriz_augmented=apply_random_augmentation(matriz)
        newname=choosen_file.replace(".txt",f"_aug{i}_s.txt")
        with open(os.path.join(root_path,newname),"w") as f:
            for row in matriz_augmented:
                f.write(" ".join(str(x) for x in row) + "\n")

def right_augmentation_2(suppine_quantity_:int,right:int,root_path:str):
    n=suppine_quantity_-right
    print(n)
    files=[file for file in sorted(os.listdir(root_path))]
    for i in range(n):
    #    print(n)
        choosen_file=rnd.choice(files)
        print(choosen_file)
        matriz=load_image_from_txt(os.path.join(root_path,choosen_file))
        matriz_augmented=apply_random_augmentation(matriz)
        newname=choosen_file.replace(".txt",f"_aug{i}_r.txt")
        with open(os.path.join(root_path,newname),"w") as f:
            for row in matriz_augmented:
                f.write(" ".join(str(x) for x in row) + "\n")

def left_augmentation(suppine_quantity_:int,left:int,root_path:str):
    n=suppine_quantity_-left
    print(n)
    files=[file for file in sorted(os.listdir(root_path))]
    for _ in range(n):
    #    print(n)
        choosen_file=rnd.choice(files)
        print(choosen_file)
        chosen=read_files(os.path.join(root_path,choosen_file))
        if chosen== None:
            raise ValueError("NO esta bien el archivo")
        array_augmented=generate_augmented_image(chosen)
        line= " ".join(str(x) for x in array_augmented)+'\n'
        with open(os.path.join(root_path,choosen_file),"a") as f:
            f.write(line)
def right_augmentation(suppine_quantity_:int,right:int,root_path:str):
    n=suppine_quantity_-right
    print(n)
    files=[file for file in sorted(os.listdir(root_path))]
    for _ in range(n):
    #    print(n)
        choosen_file=rnd.choice(files)
        print(choosen_file)
        chosen=read_files(os.path.join(root_path,choosen_file))
        if chosen== None:
            raise ValueError("NO esta bien el archivo")
        array_augmented=generate_augmented_image(chosen)
        line= " ".join(str(x) for x in array_augmented)+'\n'
        with open(os.path.join(root_path,choosen_file),"a") as f:
            f.write(line)
def read_files(root_path:str)->str|None:
    choosen=None 
    with open(root_path,"r") as f:
        for i,linea in enumerate(f,start=1):
            if rnd.randint(1,i)==1:
                choosen=linea
    return choosen
def generate_augmented_image(line):
    matriz=resize_image(line)
    arr_aug=apply_random_augmentation(matriz)
    return arr_aug.reshape(-1)

def resize_image(flatten_image:str):
    flatten_image_list=flatten_image.strip().split()
    return np.array(flatten_image_list,dtype=np.float32).reshape(32,64)

