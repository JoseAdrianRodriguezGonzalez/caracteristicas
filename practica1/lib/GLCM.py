
import cv2
import numpy as np

class GLCM:
    def __init__(self, images_path: list[str], window_size=3, levels=256):
        self.path = images_path
        self.images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in images_path]
        self.window_size = window_size
        self.levels = levels
        
        # Resultados
        self.Energia = []
        self.Contraste = []
        self.Correlacion = []
        self.Homogeneidad = []
        self.IDF_ = []
        self.Entropia = []
        self.Varianza = []

    # ---------------- Ángulo a desplazamiento ----------------
    def conversion_angulo(self, distancia, angulo):
        if angulo == 0:
            return distancia, 0
        elif angulo == 45:
            return distancia, -distancia
        elif angulo == 90:
            return 0, distancia
        elif angulo == 135:
            return -distancia, -distancia
        else:
            raise ValueError("Ángulo no válido: use 0, 45, 90 o 135")

    # ---------------- Construir GLCM local ----------------
    def glcm_matrix(self, imagen, distancia, angulo):
        dx, dy = self.conversion_angulo(distancia, angulo)
        K, L = imagen.shape
        pad = self.window_size // 2
        padded = np.pad(imagen, pad, mode='edge')
        
        P_total = np.zeros((self.levels, self.levels), dtype=np.float32)
        
        for i in range(K):
            for j in range(L):
                W = padded[i:i+self.window_size, j:j+self.window_size]
                # Coordenadas vecinas
                ni, nj = pad + dy, pad + dx
                ni = np.clip(ni, 0, self.window_size-1)
                nj = np.clip(nj, 0, self.window_size-1)
                
                val_centro = W[pad, pad]
                val_vecino = W[ni, nj]
                P_total[val_centro, val_vecino] += 1
                P_total[val_vecino, val_centro] += 1  # simétrica
        
        if P_total.sum() > 0:
            P_total /= P_total.sum()
        return P_total

    # ---------------- Funciones de métricas ----------------
    def energia_calc(self, P):
        return np.sum(P**2)

    def contraste_calc(self, P):
        i, j = np.indices(P.shape)
        return np.sum((i-j)**2 * P)

    def correlacion_calc(self, P):
        i, j = np.indices(P.shape)
        px = np.sum(P, axis=1)
        py = np.sum(P, axis=0)
        mu_x = np.sum(i[:,0]*px)
        mu_y = np.sum(j[0,:]*py)
        sigma_x = np.sqrt(np.sum(((i[:,0]-mu_x)**2)*px))
        sigma_y = np.sqrt(np.sum(((j[0,:]-mu_y)**2)*py))
        if sigma_x == 0 or sigma_y == 0:
            return 0.0
        return np.sum(((i - mu_x)*(j - mu_y)*P) / (sigma_x*sigma_y))

    def homogeneidad_calc(self, P):
        i, j = np.indices(P.shape)
        return np.sum(P / (1.0 + np.abs(i-j)))

    def IDF_calc(self, P):
        i, j = np.indices(P.shape)
        return np.sum(P / (1.0 + (i-j)**2))

    def entropia_calc(self, P):
        P_nonzero = P[P>0]
        return -np.sum(P_nonzero * np.log2(P_nonzero))

    def varianza_calc(self, P):
        i, j = np.indices(P.shape)
        mu_x = np.sum(i[:,0]*np.sum(P, axis=1))
        return np.sum(((i - mu_x)**2) * P)

    # ---------------- Computar todas las métricas para todas las imágenes ----------------
    def compute_all(self, distancia=1, angulo=0):
        for img in self.images:
            P = self.glcm_matrix(img, distancia, angulo)
            self.Energia.append(self.energia_calc(P))
            self.Contraste.append(self.contraste_calc(P))
            self.Correlacion.append(self.correlacion_calc(P))
            self.Homogeneidad.append(self.homogeneidad_calc(P))
            self.IDF_.append(self.IDF_calc(P))
            self.Entropia.append(self.entropia_calc(P))
            self.Varianza.append(self.varianza_calc(P))

