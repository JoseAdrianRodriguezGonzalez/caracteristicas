"""
En este archivo se hace el aumentation de los datos
"""
import os
import random
from lib.filters import *
def apply_filter(input_path:str,output_path:str,quantity:int)->bool:
    full_images=[image for image in os.listdir(input_path)]
    random.seed(42)
    samples=random.choices(full_images,k=quantity)
    filtersAll = [
        (gaussian_noise, {'mean':0, 'std':random.randint(5,10)}),
        (gaussian_blur, {'ksize':random.choice([3,5])}),
        (contrast, {'alpha':random.uniform(0.9,1.2), 'beta':random.randint(-10,10)}),
        (brightness, {'value':random.randint(-20,20)}),
        (saturation, {'satur':random.uniform(0.8,1.2)}),
        (rotation, {'rotation':random.choice([-10,-5,5,10])}),
        (flip, {'mode':random.choice([0,1,-1])}),
        (blur, {'ksize':random.choice([3,5])})
    ]
    count=0
    for img in samples:
        input_img=os.path.join(input_path,img)
        image=cv2.imread(input_img)
        f,params=random.choice(filtersAll)
        img_aug=f(image,**params)
        cv2.imwrite(os.path.join(output_path,f"aug_{count}_{img}"),img_aug)
        count+=1
    if len(samples)==count:
        print(f"La imagenes fueron aumentadas correctamente")
        return True
    print(f"No todas las imagenes fueron procesadas, faltaron {len(samples)-count}")
    return False

