import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
from samples import Samples
def return_images(path:str)->list[str]:
    dataset=Samples(path,crop=60)
    _,img=dataset.read_images()
    return img
def return_filter(image:list[str],decision=True)->list[str]:
    if decision:
        return list(filter(lambda x: "patch" in x,image))
    return list(filter(lambda x: "patch" not in x and "Mask" not in x,image))
