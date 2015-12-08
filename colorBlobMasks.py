
import cv2
import numpy as np
from numpy import linalg as LA
import sys
from matplotlib  import pyplot as plt
import hsvFilter_m as hf
from operator import attrgetter
windowX,windowY = 0,0 #Default window location

# a simple and short routine to show an image using namedWindow
def showMe(name,image):
    global windowX,windowY
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name,image)
    #cv2.moveWindow(name,windowX,windowY)
    #Get image shape and update window location so windows don't come up on top of eachother
    tup = image.shape
    windowX= windowX+tup[1] +4
    if windowX > 1366: #Restart at zero when windows start to run off screen **Note**screen resolution is not always 1366x768
        windowX=0
        windowY= windowY+tup[0]
        if windowY > 784: #Kunal...should this be 768 instead of 784? In general, smallest screen resolution will be 1366x768.
            windowY=0

# readThresholds reads the HSV thresholds saved in a text file
def readThresholds(fileName):
	minHSV = np.array([-1,-1,-1]) #Create arrays to store thresholds
	maxHSV = np.array([-1,-1,-1])
	try :
	    f1 = open(fileName,"r")
	    minHSV[0] = int(f1.readline()) #read min H
	    minHSV[1] = int(f1.readline()) #read min S
	    minHSV[2] = int(f1.readline()) #read min V
	    maxHSV[0] = int(f1.readline())
	    maxHSV[1] = int(f1.readline())
	    maxHSV[2] = int(f1.readline())
	    #print minHSV , maxHSV
	    f1.close()
	    return (minHSV,maxHSV)
	except IOError:
	    print 'Unable to open the threshold file\n'
	    cv2.destroyAllWindows()
	    sys.exit()
       
#****************Main Code******************************************
# Threshold filenames
blueThreshFile = 'thresholds/blueOut.txt'
yellowThreshFile = 'thresholds/yellowOut.txt'
#blueThreshFile = 'blueHome.txt'
#yellowThreshFile = 'yellowHome.txt'

cap = cv2.VideoCapture(0)

print('Press q to exit')
while cv2.waitKey(1)& 0xFF != ord('q'): 
    ret,myImg = cap.read()   
    myImg = cv2.medianBlur(myImg,3)
    cv2.imshow('orig',myImg)
    
    blues,maskList = hf.getHSVFilters(myImg,readThresholds(blueThreshFile),(11,11),2000000)
    
    if len(maskList) >0 : 
        sorted(maskList,key = lambda m: m[1][0]['m00'] ,reverse=True)
        M,center,vecta2,vectb1,vectb2,angle,Vect,L= maskList[0][1]	
        rotMat = cv2.getRotationMatrix2D( center, angle, 1.0 )    
        blues = cv2.warpAffine(blues, rotMat, (blues.shape[0],blues.shape[1]))
        myImg = cv2.warpAffine(myImg, rotMat, (blues.shape[0],blues.shape[1]))
        cv2.imshow('filterB',blues)
        yellows = hf.getHSVFilters(myImg,readThresholds(yellowThreshFile),(5,5),20000)
        cv2.imshow('filterY',yellows[0])

cap.release()   
cv2.destroyAllWindows()
sys.exit()

  
