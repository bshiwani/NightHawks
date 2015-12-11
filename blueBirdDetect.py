
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
    #cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
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
    minHSV = np.array([0,0,0]) #Create arrays to store thresholds setting default threshold values
    maxHSV = np.array([179,255,255])
    ignoreV = True #Ignore brightness threshold
    try :
        f1 = open(fileName,"r")
        minHSV[0] = int(f1.readline()) #read min H
        minHSV[1] = int(f1.readline()) #read min S
        minHSV[2] = int(f1.readline()) #read min V
        maxHSV[0] = int(f1.readline())
        maxHSV[1] = int(f1.readline())
        maxHSV[2] = int(f1.readline())
        
        if(ignoreV): #Restore to default values if set to ignore brightness
            minHSV[2] = 0
            maxHSV[2] = 255
            
        #print(minHSV,maxHSV)
        f1.close()
        return (minHSV,maxHSV)
    except IOError:
        print 'Unable to open the threshold file\n'
        cv2.destroyAllWindows()
        sys.exit()

#Checks if two masks are adjacent
def adjacent(mask1,mask2):
    kernel = np.ones((5,5), dtype = np.uint8)
    res = cv2.bitwise_or(mask1,mask2)
    res = cv2.dilate(res,kernel,iterations =1)
    res = cv2.erode(res,kernel,iterations =1)
    contours, hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #Count child contours
    childContours = 0
    for contour in hierarchy[0]:
        if contour[3] != -1:
            childContours += 1
    #Check if there is only one contour at top level
            #adjacent masks will have only one contour at top level
    if((len(contours)-childContours)==1):
        return(True,res)
    return(False,None)
    
       
#****************Main Code******************************************
# Threshold filenames
blueThreshFile = 'thresholds/blueOut.txt'
yellowThreshFile = 'thresholds/yellowOut.txt'

#Read thresholds from files:
blueThresh = readThresholds(blueThreshFile)
yellowThresh = readThresholds(yellowThreshFile)

#Initialize video capture object
cap = cv2.VideoCapture(0)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8), np.uint8) #Use ellipse structuring element because blob shape is not boxy.

#Start looping
print('Press q to exit')
while cv2.waitKey(1)& 0xFF != ord('q'): 
    ret,myImg = cap.read()   
    myImg = cv2.medianBlur(myImg,3)
    
    #Get blue masks
    blues,maskList = hf.getHSVFilters(myImg,blueThresh,(5,5),200000)
    showMe("blues", blues)
    
    #Get yellow masks
    yellows,yellowMaskList = hf.getHSVFilters(myImg,yellowThresh,(1,1),20000)
    #print("blueMaskListLength", len(maskList))
    #print("yellowMaskListLength", len(yellowMaskList))
    showMe("yellows", yellows)
    
    #if len(yellowMaskList):
        #showMe('yellows', yellowMaskList[0][0])
    
    #For every blue mask, check if there are any adjacent yellow masks, then check
    #for other adjacent blue masks (head that sometimes gets cut off)
    for i in xrange(len(maskList)):
        contiguousYellows = 0 #The number of detected adjacent yellow blobs
        birdMask = np.zeros((np.shape(myImg)[0], np.shape(myImg)[1]), dtype = np.uint8) #birdMask is the sum of all detected adjacent masks
        
        #Check for adjacent yellow masks to selected blue mask:
        for yellowMask in yellowMaskList:
            (adj,res) = adjacent(maskList[i][0],yellowMask[0])
            
            if(adj):
                #Get moments to compare areas
                Mb = cv2.moments(maskList[i][0])
                My = cv2.moments(yellowMask[0])
                #Check if yellow area less than blue area
                if(My['m00']<Mb['m00']):
                    birdMask = cv2.bitwise_or(birdMask,res)
                    contiguousYellows += 1

        #If we've detected adjacent yellow masks, check for detached head
        #Note this is not a necessary condition for detection, but adding head will place box in correct place
        if(contiguousYellows >= 2):
            for j in xrange(len(maskList)):
                if(j != i):
                    (adj,res) = adjacent(birdMask,maskList[j][0])
                    if(adj):
                        #Get moments to compare areas
                        Mb = cv2.moments(maskList[i][0])
                        Mh = cv2.moments(maskList[j][0])
                        #Check if possible head area less than original blue area
                        if(Mh['m00']<Mb['m00']):
                            birdMask = cv2.bitwise_or(birdMask,res)
            
        if(contiguousYellows >= 2):
            _,birdMask = cv2.threshold(birdMask*255,127,255,cv2.THRESH_BINARY)
            showMe('Blue Bird', birdMask)
            contours, hierarchy = cv2.findContours(birdMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0])
            cv2.rectangle(myImg,(x,y),(x+w,y+h),(255,0,0),2)
            break
        
    showMe('orig',myImg)

cap.release()   
cv2.destroyAllWindows()
sys.exit()

  
