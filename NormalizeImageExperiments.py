# -*- coding: utf-8 -*-
"""
@author: jeffrey
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#Read the image
img = cv2.imread('bird1dark.jpg',1)
if(img == None):
    print("File read error")
#resize image to have a smaller resolution
rows,cols,channels = img.shape
newRes = 400000 #The new resolution in pixels (rough)
k = float(newRes)/(rows*cols)
img = cv2.resize(img,(int(math.sqrt(k)*cols), int(math.sqrt(k)*rows)))

#Normalize BGR image

(b,g,r) = cv2.split(img)

# CLAHE (contrast limited adaptive histogram equalization). 
#Prevents loss of information in washed out areas of the imag, but seems to 
#add noise to it.

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
b_norm = clahe.apply(b)
g_norm = clahe.apply(g)
r_norm = clahe.apply(r)

'''
#Or, use regular histogram equalization. Results not as good, but less noisy
b_norm = cv2.equalizeHist(b)
g_norm = cv2.equalizeHist(g)
r_norm = cv2.equalizeHist(r)
'''


img_norm = cv2.merge((b_norm,g_norm,r_norm))

#Apply gaussian blur to minimize noise from CLAHE
#img_norm = cv2.GaussianBlur(img_norm, (5,5), 0)


#Remove all color information from image (normalize V to nominal value)
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(HSV)
V = 150*np.ones(np.shape(V),np.uint8)
img_norm1 = cv2.merge((H,S,V))
img_norm1 = cv2.cvtColor(img_norm1, cv2.COLOR_HSV2BGR)

#Same but with CLAHE preprocessing
HSV = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(HSV)
V = 150*np.ones(np.shape(V),np.uint8)
img_norm = cv2.merge((H,S,V))
img_norm = cv2.cvtColor(img_norm, cv2.COLOR_HSV2BGR)

#show the two plots side by side
plt.figure(1)
plt.subplot(121)
plt.axis('off')
plt.title('Original')
plt.imshow(cv2.cvtColor(img_norm1,cv2.COLOR_BGR2RGB))

plt.subplot(122)
plt.axis('off')
plt.title('Normalized')
plt.imshow(cv2.cvtColor(img_norm,cv2.COLOR_BGR2RGB))

plt.show()
