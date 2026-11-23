import pandas as pd 
def read_dataset(name_file:str)->pd.DataFrame:
    df=pd.read_csv(name_file,sep=" ",header=None)
    print(df.shape)
    return df
def read_output(name_file:str)->pd.DataFrame:
    df=pd.read_csv(name_file)
    print(df.shape)
    return df

def columns_naming(df:pd.DataFrame)->pd.DataFrame:
    colnames=["index"]
    for i in range(1,26):
        colnames+=[f"Joint{i}_x",f"Joint{i}_y",f"Joint{i}_z"]
    df.columns=colnames
    
    return df
