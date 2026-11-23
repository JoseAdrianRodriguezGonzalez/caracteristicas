import pandas as pd
from entrypoint.read import read_dataset

class Filters:
    def __init__(self,size_window:int,file_name:str) -> None:
        self.size_window=size_window
        self.file_name=file_name
        self.data=read_dataset(file_name)
        print(self.data.shape)
    def filtro_medio(self):
        coords=self.data.iloc[:,1:]
        filtered=coords.rolling(window=self.size_window,center=True).mean()
        filtered.insert(0,"index",self.data.iloc[:,0])
        return filtered.dropna()
    """
        version lenta 
    def media(self,frames)->float:
        return frames.sum(axis=0)/self.size_window
    def filtro_medio(self):
        middle=self.size_window//2
        for cols in self.data.columns[1:]:
            for index,_ in enumerate(self.data[cols].values):
                self.data.at[middle+index,cols]=self.media(self.data[cols][index:self.size_window+index])
        return self.data
    """
