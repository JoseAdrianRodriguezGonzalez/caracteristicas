import pandas as pd 
import numpy as np 
def create_csv(lista_f,lista_l,path):
    header=[key for key in lista_f[0].keys()]
    data={i:[] for i in header}
    for row in lista_f:
        for keys,items in row.items():
            data[keys].append(items)
    df=pd.DataFrame(data)
    df["labels"]=lista_l
    df.to_csv(path)
        

