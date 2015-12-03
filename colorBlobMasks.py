
import cv2
import numpy as np
from numpy import linalg as LA
import sys
from matplotlib  import pyplot as plt
import hsvFilter_m as hf
windowX,windowY = 0,0

# a simple and short routine to show an image using namedWindow
def showMe(name,image):
    global windowX,windowY
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name,image)
    #cv2.moveWindow(name,windowX,windowY)
    tup = image.shape
    windowX= windowX+tup[1] +4
    if windowX > 1366:
        windowX=0
        windowY= windowY+tup[0]
        if windowY > 784:
            windowY=0

minHSV = np.array([-1,-1,-1]) 
maxHSV = np.array([-1,-1,-1])
try :
    f1 = open(str(sys.argv[1]),"r")
    minHSV[0] = int(f1.readline())
    minHSV[1] = int(f1.readline())
    minHSV[2] = int(f1.readline())
    maxHSV[0] = int(f1.readline())
    maxHSV[1] = int(f1.readline())
    maxHSV[2] = int(f1.readline())
    #print minHSV , maxHSV
    f1.close()
except IOError:
    print 'unable to open threshold file\n'
    cv2.destroyAllWindows()
    sys.exit()
       
cap = cv2.VideoCapture(0)
print 'press q to exit'
while cv2.waitKey(1)& 0xFF != ord('q'):
    ret,myImg = cap.read()
   
    myImg = cv2.medianBlur(myImg,3)
    cv2.imshow('orig',myImg)
    cv2.imshow('filter1',hf.getHSVFilters(myImg,(minHSV,maxHSV),(11,11))[0])
cap.release()   
cv2.destroyAllWindows()
sys.exit()

  
