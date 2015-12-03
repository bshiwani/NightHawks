 
import cv2
import numpy as np
from numpy import linalg as LA
import sys
from matplotlib  import pyplot as plt

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

class thresholding:

    minHSV = np.array([-1,-1,-1]) # stores lower bound of HSV values
    maxHSV = np.array([-1,-1,-1]) #stores upper bound of HSV
    H=np.zeros((2,2),np.uint8)
    S=np.zeros((2,2),np.uint8)
    V=np.zeros((2,2),np.uint8)
    HSV = np.zeros((2,2,1),np.uint8)

    def thresholding():
	self.minHSV = np.array([-1,-1,-1]) # stores lower bound of HSV values
    	self.maxHSV = np.array([-1,-1,-1]) #stores upper bound of HSV
	self.H=np.zeros((2,2),np.uint8)
	self.S=np.zeros((2,2),np.uint8)
	self.V=np.zeros((2,2),np.uint8)
	self.HSV = np.zeros((2,2,1),np.uint8)
        return self

    # routine to update threshold
    def updateHSVlimits(self,x,y):
        if self.minHSV[0] == -1:
            self.minHSV[0] = self.H[x,y]
            self.minHSV[1] = self.S[x,y]
            self.minHSV[2] = self.V[x,y]
        if self.maxHSV[0] == -1:
            self.maxHSV[0] = self.H[x,y]
            self.maxHSV[1] = self.S[x,y]
            self.maxHSV[2] = self.V[x,y]
                
        if self.minHSV[0] >= self.H[x,y]:
            self.minHSV[0] = self.H[x,y]
        if self.minHSV[1] >= self.S[x,y]:
            self.minHSV[1] = self.S[x,y]
        if self.minHSV[2] >= self.V[x,y]:
            self.minHSV[2] = self.V[x,y]
            
        if self.maxHSV[0] <= self.H[x,y]:
            self.maxHSV[0] = self.H[x,y]
        if self.maxHSV[1] <= self.S[x,y]:
            self.maxHSV[1] = self.S[x,y]
        if self.maxHSV[2] <= self.V[x,y]:
            self.maxHSV[2] = self.V[x,y]
            
    def mouseCallBack(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONUP:
            self.updateHSVlimits(y,x)
            myFilt = cv2.inRange(self.HSV,self.minHSV,self.maxHSV)
            M = cv2.moments(myFilt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            covar = np.matrix([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])
            cv2.circle(myFilt,(cx,cy),5,(100,0,0),-1)
            cv2.imshow('Threshold Result',myFilt)
    
    def startRecordingThreshold(self,inputImage):
        original = inputImage
        showMe('Recording Threshold',original)
        # get HSV and seperate images
        self.HSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        self.H,self.S,self.V = cv2.split(self.HSV)
        cv2.setMouseCallback('Recording Threshold',self.mouseCallBack)

    def saveThresholds(self,outputFile):
        f1 = open(outputFile,"w")
        f1.write(str(self.minHSV[0])+"\n")
        f1.write(str(self.minHSV[1])+"\n")
        f1.write(str(self.minHSV[2])+"\n")
        f1.write(str(self.maxHSV[0])+"\n")
        f1.write(str(self.maxHSV[1])+"\n")
        f1.write(str(self.maxHSV[2])+"\n")
        f1.close()
        cv2.destroyWindow('Recording Threshold')
        cv2.destroyWindow('Threshold Result')


   
