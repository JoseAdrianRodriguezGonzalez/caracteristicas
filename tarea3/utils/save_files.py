import pandas as pd 
import os
def save_file(df:pd.DataFrame,file_name:str):
    os.makedirs("out/",exist_ok=True)
    df.to_csv("out/"+file_name,index=False)
