import cv2
import numpy as np
from numpy import linalg as LA
import sys
from math import sqrt
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
            
def maskColorCheck(HSVImage, testMask, testMaskArea, thresholdArrayTuppleHSV):
    (minHSV,maxHSV) = thresholdArrayTuppleHSV
    #apply contour mask to input image
    img = HSVImage*cv2.merge([testMask, testMask, testMask])
    img = cv2.inRange(HSVImage, minHSV, maxHSV)
    
    #get inRange area
    M = cv2.moments(img)
    
    #compare inRange area to total area. If high, it is correct color
    if(M['m00']/testMaskArea >0.99):
        return True
    return False
    
    

    

def getHSVFilters(inputImage, thresholdArrayTuppleHSV, kernelSize, minimumArea):
    (minHSV,maxHSV) = thresholdArrayTuppleHSV

    orig = inputImage.copy() #make a copy of the input image
    myFilt = np.zeros(orig.shape) #Do we need to initialize this???
    
    # Convert to HSV and find regions that are in range defined by thresholds
    HSV = cv2.cvtColor(orig,cv2.COLOR_BGR2HSV)
    myFilt = cv2.inRange(HSV,minHSV,maxHSV)
    #showMe("FILTERED",myFilt)
    
    #Apply morphological operations on the in range pixels (clean up noise)
    #kernel = np.ones(kernelSize,np.uint8) #replaced with ellipse below
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize) #Use ellipse structuring element because blob shape is not boxy.
    myFilt = cv2.morphologyEx(myFilt,cv2.MORPH_CLOSE,kernel)
    myFilt = cv2.morphologyEx(myFilt,cv2.MORPH_OPEN,kernel)
    
    #myFilt = cv2.dilate(myFilt,kernel,iterations =1) #Dilation followed by erosion is closing...replaced with close operation
    #myFilt = cv2.erode(myFilt,kernel,iterations =1)
    #showMe("Morph",myFilt) #Debug
    
    # dist = cv2.distanceTransform(np.array(myFilt,np.uint8)*255,cv2.cv.CV_DIST_FAIR,5)
    #showMe("Distance", dist+myFilt)

    #Find the contours of the in range pixels
    contours,hirarchy = cv2.findContours(myFilt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    maskList = []
    while i <(len(contours)):
        hull = cv2.convexHull(contours[i])
        mask = np.zeros(myFilt.shape,np.uint8) #Create an empty mask
        cv2.drawContours(mask,contours,i,(255,0,0),thickness = cv2.cv.CV_FILLED) #Draw each contour as a filled mask
        if(hirarchy[0][i][3] == -1):
            M = cv2.moments(mask) #Get the moments of the mask to check if it is above minimum area
            #showMe('mask', mask)
            if M['m00'] >minimumArea :
                if(maskColorCheck(HSV, mask, M['m00'], thresholdArrayTuppleHSV)):
                    
                    M,vecta1,vecta2,vectb1,vectb2,angle,Vect,L = getShapeProperties(mask)
                    mask = mask/255 #Create binary image mask
                    
                    #mask = cv2.merge((mask,mask,mask)) #Make a three channel mask for multiplying with 3 channel image
                    #masked = mask*orig #Get masked sections of original image
                    
                    #showMe('Masked',masked);
                    cv2.circle(myFilt,vecta1,5,(100,0,0),-1)
                    #angle =180 - np.rad2deg(np.arctan2(pow(M['mu30'],1),pow(M['mu03'],1))) 	
                    cv2.polylines(orig,[np.array(hull,np.int32)],True,(0,100,100))
                    cv2.line(orig,vecta1,vecta2,(0,255,0),5)
                    cv2.line(orig,vectb1,vectb2,(255,0,0),5)
                    maskList.append((mask,(M,vecta1,vecta2,vectb1,vectb2,angle,Vect,L))) #Add each mask and its properties to the maskList	
        i = i+1
    #showMe("Hull",orig)
    return orig,maskList #Return (original image with blob polylines and eigenvectors), and (the list of individual blob masks) 

def getShapeProperties(binaryMask):
    M = cv2.moments(binaryMask)
    cx = int(M['m10']/M['m00']) 
    cy = int(M['m01']/M['m00'])
    covar = np.matrix([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])
    L, Vect = LA.eig(covar)
    #try:
    scalingFactora = L[0]/(L[0]+L[1])*sqrt(M['m00'])/10
    scalingFactorb = L[1]/(L[0]+L[1])*sqrt(M['m00'])/10

    vecta1 = (int(cx),int(cy))
    vectb1 = (int(cx),int(cy))
    
    xa = int( -scalingFactora *Vect[0,0])#* pow(M['m30'],1/3)
    ya = int( -scalingFactora *Vect[1,0])#* pow(M['m03'],1/3)   
    
    xb = int(scalingFactorb*Vect[0,1])# * pow(M['m30'],1/3)
    yb = int(scalingFactorb*Vect[1,1])# * pow(M['m03'],1/3)
    

    vecta2 = (vecta1[0]+xa,vecta1[1]+ya)
    vectb2 = (vectb1[0]+xb,vectb1[1]+yb)	
    
    if L[1]<L[0]:
	if((M['mu30']-M['mu03'])>0):
	    shift = -90
	    angle = 90-np.rad2deg(np.arctan2(xb,yb))
	else:
	    shift = 90
	    angle = 270-np.rad2deg(np.arctan2(xb,yb))		
   
    else :
	if((M['mu30']-M['mu03'])>0):
	    shift = -90
	    angle = 90-np.rad2deg(np.arctan2(xa,ya)) 
	else:
	    shift = 90
	    angle = 270 -np.rad2deg(np.arctan2(xa,ya)) 
    #angle =180 - np.rad2deg(np.arctan2(pow(M['mu30'],1),pow(M['mu03'],1))) 
    return M,vecta1,vecta2,vectb1,vectb2,angle,Vect,L 
		#masked,M,vecta1,vecta2,vectb1,vectb2,angle,vect,L
    	
