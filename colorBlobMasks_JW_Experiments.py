
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
       
#****************Main Code******************************************
# Threshold filenames
blueThreshFile = 'thresholds/blueOut.txt'
yellowThreshFile = 'thresholds/yellowOut.txt'
#blueThreshFile = 'blueHome.txt'
#yellowThreshFile = 'yellowHome.txt'

#Read thresholds from files:
blueThresh = readThresholds(blueThreshFile)
yellowThresh = readThresholds(yellowThreshFile)

#Initialize video capture object
cap = cv2.VideoCapture(0)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8), np.uint8) #Use ellipse structuring element because blob shape is not boxy.
kernel = np.ones((10,10), dtype = np.uint8)
#Start looping
print('Press q to exit')
while cv2.waitKey(1)& 0xFF != ord('q'): 
    ret,myImg = cap.read()   
    myImg = cv2.medianBlur(myImg,3)
    showMe('orig',myImg)
    
    
    #Get blue masks
    blues,maskList = hf.getHSVFilters(myImg,blueThresh,(5,5),200000)
    cv2.imshow("blues", blues)
    
    #Get yellow masks
    yellows,yellowMaskList = hf.getHSVFilters(myImg,yellowThresh,(1,1),20000)
    #print("blueMaskListLength", len(maskList))
    #print("yellowMaskListLength", len(yellowMaskList))
    cv2.imshow("yellows", yellows)
    
    #if len(yellowMaskList):
        #showMe('yellows', yellowMaskList[0][0])
    
    for i in xrange(len(maskList)):#For every blue mask
        contiguousYellows = 0
        birdMask = np.zeros((np.shape(myImg)[0], np.shape(myImg)[1]), dtype = np.uint8)
        
        for yellowMask in yellowMaskList:#Check every yellow mask to see if it is contiguous with blue mask
            #res = maskList[i][0]+yellowMask[0]
            res = cv2.bitwise_or(maskList[i][0],yellowMask[0])
            #res = cv2.morphologyEx(res,cv2.MORPH_CLOSE,kernel)
            resD = cv2.dilate(res,kernel,iterations =1) 
            #res = cv2.erode(res,kernel,iterations =1)
            #showMe('totalMask', res*255)
            contours, hierarchy = cv2.findContours(resD,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #print(len(contours))
            if(len(contours) == 1):
                birdMask = cv2.bitwise_or(birdMask,res)
                contiguousYellows += 1
                
                #showMe('birdmask', birdMask*255)
                #x,y,w,h = cv2.boundingRect(birdMask)
                #cv2.rectangle(myImg,(x,y),(x+w,y+h),(0,255,0),2)
        #showMe('birdMask', birdMask)
                
            #If we've added yellows, check for blues that are also contiguous (add head)
            for j in xrange(len(maskList)):
                if(j != i):
                    #res1 = maskList[j][0] + birdMask
                    res1 = cv2.bitwise_or(maskList[j][0],birdMask)
                    #res1 = cv2.morphologyEx(res1,cv2.MORPH_CLOSE,kernel)
                    resD1 = cv2.dilate(res1,kernel,iterations =1) 
                    #res1 = cv2.erode(res1,kernel,iterations =1)
                    #contours, hierarchy = cv2.findContours(res1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    contours, hierarchy = cv2.findContours(resD1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    #print(len(contours))
                    if(len(contours) == 1):
                        #birdMask = birdMask + res1
                        birdMask = cv2.bitwise_or(birdMask,res1)
        if contiguousYellows >= 2:
            showMe('birdMask', birdMask*255)
            break
    
                        

    
    
            
            
    
    '''
    if len(maskList) >0 : #If there are blue objects found:
    
        #sorted(maskList,key = lambda m: m[1][0]['m00'] ,reverse=True) #Sort maskList according to size of blob
        #M,center,vecta2,vectb1,vectb2,angle,Vect,L= maskList[0][1]	
        #showMe('maskList 0', maskList[0][0]*255)
        yellows, yellowMaskList = hf.getHSVFilters(myImg,yellowThresh,(5,5),20000)
        
        
        #sorted(yellowMaskList, key = lambda m: m[1][0]['m00'], reverse = True) #Sort
        
        rotMat = cv2.getRotationMatrix2D( center, angle, 1.0 )    
        #blues = cv2.warpAffine(blues, rotMat, (blues.shape[0],blues.shape[1]))
        #myImg = cv2.warpAffine(myImg, rotMat, (blues.shape[0],blues.shape[1]))
        
        cv2.imshow('filterB',blues)
        yellows = hf.getHSVFilters(myImg,yellowThresh,(5,5),20000)
        cv2.imshow('filterY',yellows[0])
        '''
    
    
    
    
    

cap.release()   
cv2.destroyAllWindows()
sys.exit()

  
