import os
def list_files(path:str)->list[str]:
    directorios=[f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) and (len(f)<=3 and "S" in f)]
    print(directorios)
    return directorios 
def labeling(path:str,waiter:list[str],left,right,suppine):
    os.makedirs(os.path.join(path,"left"),exist_ok=True)
    os.makedirs(os.path.join(path,"right"),exist_ok=True)
    os.makedirs(os.path.join(path,"suppine"),exist_ok=True)
    counter=1
    for person in sorted(waiter):
        if "left" in person or "right" in person or "suppine" in person:
            print("si encontro y me salto")
            continue
        for file in sorted(os.listdir(os.path.join(path,person))):
            #print(os.path.join(path,person,file)) 
            counter+=1
            number=file[:-4]
            if number in left:
                os.rename(os.path.join(path,person,file),os.path.join(path,"left/",str(counter)+"_"+file))
            elif number in right:
                os.rename(os.path.join(path,person,file),os.path.join(path,"right/",str(counter)+"_"+file))
            elif number in suppine:
                os.rename(os.path.join(path,person,file),os.path.join(path,"suppine/",str(counter)+"_"+file))
            print(number)
def count_samples_txt(folder):
    total=0
    for fname in os.listdir(folder):
        path=os.path.join(folder,fname)
        if os.path.isfile(path):
            with open(path,"r") as f:
                for _ in f:
                    total+=1
    return total
def count_samples_txt_2(folder): 
    return sum(1 for fname in os.listdir(folder) if os.path.isfile(os.path.join(folder,fname)))
def labeling_2(path:str,waiter:list[str],left,right,suppine):
    directorios=[ file for file in sorted(os.listdir(os.path.join(path,sorted(waiter)[0]))) if os.path.isdir(os.path.join(path,sorted(waiter)[0],file))]
    for carpeta in directorios:
        os.makedirs(os.path.join(path,carpeta),exist_ok=True)
        os.makedirs(os.path.join(path,carpeta,"left"),exist_ok=True)
        os.makedirs(os.path.join(path,carpeta,"right"),exist_ok=True)
        os.makedirs(os.path.join(path,carpeta,"suppine"),exist_ok=True)
    """
    for person in sorted(waiter):
        if "left" in person or "right" in person or "suppine" in person:
            print("si encontro y me salto")
            continue  
        print(directorios)
        for carpeta in directorios:
            for file in sorted(os.listdir(os.path.join(path,person,carpeta))):
                directorio= os.path.join(path,person,carpeta,file)
                if "Matrix" in directorio:
                    file=file[-6:]
                    print(file)
                    os.rename(directorio,os.path.join(path,person,carpeta,file))
    """
    counter=1
    for person in sorted(waiter):
        if "left" in person or "right" in person or "suppine" in person:
            print("si encontro y me salto")
            continue
        for carpeta in directorios:
            for file in sorted(os.listdir(os.path.join(path,person,carpeta))):
                number=file[:-4]
                counter+=1
                if number in left:
                    os.rename(os.path.join(path,person,carpeta,file),os.path.join(path,carpeta,"left/",str(counter)+"_"+file))
                elif number in right:
                    os.rename(os.path.join(path,person,carpeta,file),os.path.join(path,carpeta,"right/",str(counter)+"_"+file))
                elif number in suppine:
                    os.rename(os.path.join(path,person,carpeta,file),os.path.join(path,carpeta,"suppine/",str(counter)+"_"+file))
            
