import cv2
import os
import numpy as np
class LBP:
    def __init__(self,images_path:list[str]):
        self.path=images_path
        self.images=[cv2.imread(path,cv2.IMREAD_GRAYSCALE) for path in images_path]
        self.media_images=[]
        self.varianza_images=[]
        self.correlacion_images=[]
        self.contraste_images=[]
        self.homogeneidad_images=[]
        self.lbp_images=[]
    def pad_image(self,image,pad_size=1):
        return cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    def compute_lbp(self, num_size=1):
        for image in self.images:
            image = self.pad_image(image, pad_size=num_size)
            lbp_img = self.lbp_calc(image, num_size)
            self.lbp_images.append(lbp_img)
    def lbp_calc(self, image, num_size):
        K, L = image.shape
        result = np.zeros((K - 2*num_size, L - 2*num_size), dtype=np.uint8)
        for row in range(num_size, K - num_size):
            for col in range(num_size, L - num_size):
                ventana = image[row-num_size:row+num_size+1, col-num_size:col+num_size+1]
                result[row - num_size, col - num_size] = self.compactar(ventana)
        return result
    def compactar(self, ventana):
        binarios = [
            ventana[0, 0], ventana[0, 1], ventana[0, 2],
            ventana[1, 2], ventana[2, 2], ventana[2, 1],
            ventana[2, 0], ventana[1, 0]
        ]
        centro = ventana[1, 1]
        bits = [1 if v > centro else 0 for v in binarios]
        return int(''.join(str(b) for b in bits), 2)
    def media(self,num_size):
        for image  in self.lbp_images:
            media=self.media_calc(num_size,image)
            self.media_images.append(media)
    def media_calc(self,size,image):
        K,L=image.shape
        resultante=np.zeros((K - 2*size, L - 2*size), dtype=np.float32)
        for row in range(size,K-size):
            for col in range(size,L-size):
                W = image[row-size:row+size+1, col-size:col+size+1]
                mu=np.mean(W)
                resultante[row-size,col-size]=mu
        resultante_resized = cv2.resize(resultante, (30, 30), interpolation=cv2.INTER_LINEAR)
        return resultante_resized
    def varianza(self, num_size):
        for image in self.lbp_images:
            media=self.vari_calc(num_size,image)
            self.varianza_images.append(media)
    def vari_calc(self,size,image):
        K,L=image.shape
        resultante=np.zeros((K - 2*size, L - 2*size), dtype=np.float32)
        for row in range(size,K-size):
            for col in range(size,L-size):
                W = image[row-size:row+size+1, col-size:col+size+1]
                mu=np.var(W)
                resultante[row-size,col-size]=mu
        resultante_resized = cv2.resize(resultante, (30, 30), interpolation=cv2.INTER_LINEAR)
        return resultante_resized
    def contraste(self, num_size):
        for image in self.lbp_images:
            media=self.contr_calc(num_size,image)
            self.contraste_images.append(media)
    def contr_calc(self,size,image):
        K,L=image.shape
        resultante=np.zeros((K - 2*size, L - 2*size), dtype=np.float32)
        for row in range(size,K-size):
            for col in range(size,L-size):
                W = image[row-size:row+size+1, col-size:col+size+1]
                mu=np.mean(W)
                contraste=np.mean((W-mu)**2)
                resultante[row-size,col-size]=contraste
        resultante_resized = cv2.resize(resultante, (30, 30), interpolation=cv2.INTER_LINEAR)
        return resultante_resized
        
    def correlacion(self, num_size):
        for image in self.lbp_images:
            media=self.corr_calc(num_size,image)
            self.correlacion_images.append(media)
    def corr_calc(self,size,image):
        K,L=image.shape
        resultante=np.zeros((K - 2*size, L - 2*size), dtype=np.float32)
        for row in range(size,K-size):
            for col in range(size,L-size):
                W = image[row-size:row+size+1, col-size:col+size+1]
                mu=np.mean(W)
                sigma=np.std(W)
                corr=np.mean(((W[:,:-1]-mu)*(W[:,1:]-mu))/(sigma**2+1e-6))
                resultante[row-size,col-size]=corr
        resultante_resized = cv2.resize(resultante, (30, 30), interpolation=cv2.INTER_LINEAR)
        return resultante_resized
    def homogeneidad(self, num_size):
        for image in self.lbp_images:
            homo_img = self.homo_calc(num_size, image)
            self.homogeneidad_images.append(homo_img)

    def homo_calc(self, size, image):
        K, L = image.shape
        resultante = np.zeros((K - 2*size, L - 2*size), dtype=np.float32)
        for row in range(size, K-size):
            for col in range(size, L-size):
                W = image[row-size:row+size+1, col-size:col+size+1]
                homo = np.mean(1 / (1 + np.abs(W[:, :-1] - W[:, 1:]+1e-6)))
                resultante[row-size, col-size] = homo
        resultante_resized = cv2.resize(resultante, (30, 30), interpolation=cv2.INTER_LINEAR)
        return resultante_resized
