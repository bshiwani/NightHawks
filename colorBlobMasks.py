
import cv2
import numpy as np
from numpy import linalg as LA
import sys
from matplotlib  import pyplot as plt
import hsvFilter_m as hf
from operator import attrgetter
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

def readThresholds(fileName):
	minHSV = np.array([-1,-1,-1]) 
	maxHSV = np.array([-1,-1,-1])
	try :
	    f1 = open(fileName,"r")
	    minHSV[0] = int(f1.readline())
	    minHSV[1] = int(f1.readline())
	    minHSV[2] = int(f1.readline())
	    maxHSV[0] = int(f1.readline())
	    maxHSV[1] = int(f1.readline())
	    maxHSV[2] = int(f1.readline())
	    #print minHSV , maxHSV
	    f1.close()
	    return (minHSV,maxHSV)
	except IOError:
	    print 'unable to open threshold file\n'
	    cv2.destroyAllWindows()
	    sys.exit()
       

blueThreshFile = 'thresholds/blue.txt'
yellowThreshFile = 'thresholds/yellow.txt'
cap = cv2.VideoCapture(0)

print 'press q to exit'
while cv2.waitKey(1)& 0xFF != ord('q'):
    ret,myImg = cap.read()   
    myImg = cv2.medianBlur(myImg,3)
    cv2.imshow('orig',myImg)
    
    blues,maskList = hf.getHSVFilters(myImg,readThresholds(blueThreshFile),(11,11),2000000)
    
    
	 
    if len(maskList) >0 : 
	sorted(maskList,key = lambda m: m[1]['m00'] ,reverse=True)
   	center ,angle = maskList[0][2],maskList[0][3]
	rotMat = cv2.getRotationMatrix2D( center, angle, 1.0 )    
	blues = cv2.warpAffine(blues, rotMat, (blues.shape[0],blues.shape[1]))  

    cv2.imshow('filterB',blues)
    yellows = hf.getHSVFilters(myImg,readThresholds(yellowThreshFile),(5,5),200000)
    cv2.imshow('filterY',yellows[0])

cap.release()   
cv2.destroyAllWindows()
sys.exit()

  
