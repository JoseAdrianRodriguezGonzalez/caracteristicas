from scipy.ndimage import rotate
import random as rnd
import numpy as np 
def aug_rotation(img,angle_max=10):
    angle=np.random.uniform(-angle_max,angle_max)
    return rotate(img,angle,reshape=False,order=1)
def aug_gaussian_noise(img,sigma=0.02):
    noise=np.random.normal(0,sigma,img.shape)
    return img+noise
AUGMENTERS=[
    aug_rotation,
    aug_gaussian_noise
]
def apply_random_augmentation(img):
    augmenter=rnd.choice(AUGMENTERS)
    return augmenter(img)

