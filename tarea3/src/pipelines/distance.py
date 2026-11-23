import pandas as pd
import numpy as np
"""
Necesito conocer los indices joints:
joint0:1
joint7:21
join11:33
joint15:45
joint19:57
joint4:

"""    
class distance:
    def __init__(self,dataframe:pd.DataFrame,joints:list[int],reference_joint:int) -> None:
        self.df=dataframe
        self.joints=joints
        self.indexes=dataframe.iloc[:,0]
        self.reference_joint=reference_joint
    def calculate_distance(self):
        diccio={}
        ref=self.df.iloc[:,self.reference_joint:self.reference_joint+3].to_numpy()
        for joint in self.joints:
            cords=self.df.iloc[:,joint:joint+3].to_numpy()
            diccio[f"d_{joint}"]=np.linalg.norm(cords-ref,axis=1)
        result=pd.DataFrame(diccio)
        result.insert(0,"index",self.indexes)
        return result
