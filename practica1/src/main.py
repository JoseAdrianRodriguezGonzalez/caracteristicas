import sys
import os

# Agregar carpeta 'lib' al path para que Python pueda encontrar samples.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

# Ahora podemos importar la clase Samples
from samples import Samples
from dataset import generate_dataset_lbp ,generate_dataset_csv,generate_dataset_GLCM_csv
#from histograms import *
from files import return_images,return_filter
if __name__ =="__main__":
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datos'))
    images=return_images(path)
    filtered=return_filter(images)
    print(filtered)
    #generate_dataset_csv(filtered)
    #generate_dataset_GLCM_csv(filtered)
    #filtered=return_filter(images,False)
    generate_dataset_lbp(filtered)
    #THis is the process tol generate the samples and therefore the images 
   
"""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datos/'))
    dataset=Samples(path,crop=60)
    print(f"{len(dataset.images)} imágenes cargadas.")
    print(f"{len(dataset.masks)} máscaras cargadas.")
    all_patches = []
    for img, mask in zip(dataset.images, dataset.masks):
        patches = dataset.create_samples(img, mask)
        all_patches.extend(patches)
    print(f"Total de patches generados: {len(all_patches)}")
    dataset.save_samples()
    print("Patches guardados con éxito.")

"""
