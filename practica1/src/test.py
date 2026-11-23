import cv2
import numpy as np
import os
path="../datos/"
folders=[f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
image_path=os.listdir(os.path.join(path,folders[0]))[0]
image=cv2.imread(os.path.join(path,folders[0],image_path),cv2.IMREAD_GRAYSCALE))
#showing image
#cv2.imshow("Imagen",image)





#Destroy process
while True:
    key=cv2.waitKey(0) & 0xFF
    if key==27 or key ==ord('q'):
        break
 ##ESperar indefinidiamente por ua tecla cualquier
cv2.destroyAllWindows()
