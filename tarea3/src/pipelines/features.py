
import pandas as pd
import numpy as np

class Features:
    def __init__(self, distance_data: pd.DataFrame, size_window: int = 10, loss_start: int = 0, loss_end: int = 0):
        self.df = distance_data
        self.size_window = size_window
        self.loss_start = loss_start
        self.loss_end = loss_end
        self.distances = distance_data.iloc[:, 1:]
        self.index = distance_data.iloc[:, 0]

    def features_all(self):
        diccionario = {}
        for idx, col in enumerate(self.distances.columns):
            series = self.distances[col]
            diccionario[f"promedio_d{idx}"] = self._block_aggregate(series, np.mean)
            diccionario[f"varianza_d{idx}"] = self._block_aggregate(series, np.var)
            diccionario[f"minimo_d{idx}"] = self._block_aggregate(series, np.min)
            diccionario[f"maximo_d{idx}"] = self._block_aggregate(series, np.max)
        return pd.DataFrame(diccionario)

    def _block_aggregate(self, series, func):
        n = len(series)
        s = self.size_window
        result = []

        if self.loss_start > 0:
            first_block_size = s - self.loss_start
            result.append(func(series[:first_block_size]))
            start = first_block_size
        else:
            start = 0
        while start + s <= n - self.loss_end:
            result.append(func(series[start:start + s]))
            start += s
        # Último bloque (ajustado por pérdida al final)
        if start < n:
            result.append(func(series[start:]))

        
        return np.array(result)

