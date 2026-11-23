import cv2 
import numpy as np
def blur(img,ksize=5):
    """
    Hace un blur
    """
    
    return cv2.blur(img,(ksize,ksize))
def gaussian_blur(img,ksize:int=5)->np.ndarray:
    """
    Aplica un filtrop gaussiano, en media 0
        """
    return cv2.GaussianBlur(img,(ksize,ksize),0)
def rotation(img,rotation:int)->np.ndarray:
    """

    """
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),rotation,scale=1.0)
    return cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
def flip(img:np.ndarray,mode:int=1)->np.ndarray:
    return cv2.flip(img,mode)
def brightness(img,value:int=30):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    h,s,v=cv2.split(hsv)
    v=np.clip(v+value,0,255)
    hsv=cv2.merge((h,s,v))
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
def contrast(img,alpha=1.2,beta=0):
    return np.clip(alpha*img+beta,0,255).astype(np.uint8)
def saturation(img,satur:float=1.2):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    h,s,v=cv2.split(hsv)
    v=np.clip(s*satur,0,255)
    hsv=cv2.merge((h,s,v))
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
def gaussian_noise(img,mean=0,std=10):
    noise=np.random.normal(mean,std,img.shape)
    nosiy=np.clip(img+noise,0,255).astype(np.uint8)
    return nosiy

