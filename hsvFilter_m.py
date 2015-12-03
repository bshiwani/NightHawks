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

def getHSVFilters(inputImage, thresholdArrayTuppleHSV, dilationKernalSize):
    (minHSV,maxHSV) = thresholdArrayTuppleHSV
    #minHSV = np.array([-1,-1,-1]) # stores lower bound of HSV values
    #maxHSV = np.array([-1,-1,-1]) #stores upper bound of HSV
    orig = inputImage
    myFilt = np.zeros(orig.shape)
    # get HSV and seperate images
    HSV = cv2.cvtColor(orig,cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV)
    myFilt = cv2.inRange(HSV,minHSV,maxHSV)
    # showMe("FILTERED",myFilt)
    #create stucturing element 
    kernel = np.ones(dilationKernalSize,np.uint8)
    # dilate the image with that element
    myFilt = cv2.dilate(myFilt,kernel,iterations =1)
    #showMe("Dilate",myFilt)
    # dist = cv2.distanceTransform(np.array(myFilt,np.uint8)*255,cv2.cv.CV_DIST_FAIR,5)
    #showMe("Distance", dist+myFilt)
    # get borders of the blob
    border = cv2.Canny(myFilt,100,200)
    #showMe("Border",border)
    masked = np.zeros(orig.shape,np.uint8)
    contours,hirarchy = cv2.findContours(myFilt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    maskList = []
    while i <(len(contours)):
        hull = cv2.convexHull(contours[i])
        mask = np.zeros(myFilt.shape,np.uint8)
        cv2.drawContours(mask,contours,i,(255,0,0),thickness = cv2.cv.CV_FILLED)
        M = cv2.moments(mask)
        if M['m00'] >200000 :
            mask = mask/255
            mask = cv2.merge((mask,mask,mask))
            masked = masked + mask*orig
            maskList.append((masked,M['m00']))
           # showMe('Masked',masked);
            cx = int(M['m10']/M['m00']) 
            cy = int(M['m01']/M['m00'])
            cv2.circle(myFilt,(cx,cy),5,(100,0,0),-1)
            covar = np.matrix([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])
            W, Vect = LA.eig(covar)
            vect1a = (int(cx),int(cy))
            vect1b = (int(cx+ M['m00']/10000*(W[0]/(W[0]+W[1]))*(Vect[0,0])),int(cy+ M['m00']/10000*(W[0]/(W[0]+W[1]))*Vect[1,0]))

            vect2a = (int(cx),int(cy))
            vect2b = (int(cx+ M['m00']/10000*(W[1]/(W[1]+W[1]))*(Vect[0,1])),int(cy+ M['m00']/10000*(W[1]/(W[1]+W[1]))*Vect[1,1]))

            cv2.polylines(orig,[np.array(hull,np.int32)],True,(0,100,100))
            cv2.line(orig,vect1a,vect1b,(0,255,0),5)
            cv2.line(orig,vect2a,vect2b,(255,0,0),5)
        i = i+1
    #showMe("Hull",orig)
    return orig,maskList

