#experiments with HuMoments
#Code based on colorBlobMasks_JW_Experiments.py
import cv2
import numpy as np
from numpy import linalg as LA
import sys
from matplotlib  import pyplot as plt
from matplotlib.pyplot import show, plot
import hsvFilter_m as hf
from operator import attrgetter
windowX,windowY = 0,0 #Default window location

# a simple and short routine to show an image using namedWindow
def showMe(name,image):
    global windowX,windowY
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name,image)
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
    ignoreV = False #Ignore brightness threshold
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

#**************HuMoments subroutines***************************************
def animate(array):
	array=array*50	#scaling factor
	delta=10	#spcacing value for bars
	graph = 255*np.ones((700,800,3),dtype=np.uint8) #image space (white)
	for i in xrange(len(array)):
		xx=int(array[i])
		cv2.line(graph,(delta,350),(delta,350-xx),(0,255,0),50) #plotting the values for each Humoment in a different bar
		delta=delta+100	#increment for next bar
	print(array)
	showMe('BarGraph',graph)
		
# End of HuMoments subroutines     
#***************************************************

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
kernel = np.ones((10,10), dtype = np.uint8)
#Start looping
print('Press q to exit')
graph = 255*np.ones((700,800,3),dtype=np.uint8)
while cv2.waitKey(1)& 0xFF != ord('q'): 
    ret,myImg = cap.read()   
    myImg = cv2.medianBlur(myImg,3)
    showMe('orig',myImg)
    plt.show()
    
    showMe('BarGraph',graph)
    #Get blue masks
    blues,maskList = hf.getHSVFilters(myImg,blueThresh,(5,5),200000)
    cv2.imshow("blues", blues)
    
    #Get yellow masks
    yellows,yellowMaskList = hf.getHSVFilters(myImg,yellowThresh,(1,1),20000)

    
    for i in xrange(len(maskList)):#For every blue mask
        contiguousYellows = 0
        birdMask = np.zeros((np.shape(myImg)[0], np.shape(myImg)[1]), dtype = np.uint8)
#********#obtain HuMoments values*************************************************
        moments=cv2.moments(maskList[0][0])	#obtain moments first
        hu0=cv2.HuMoments(moments)	#call humoments function
        hu=-np.sign(hu0)*np.log10(np.abs(hu0))	#diminish dynamic range using logarithmic formula
        x=hu[0][0],hu[1][0],hu[2][0],hu[3][0],hu[4][0],hu[5][0],hu[6][0]
        y=np.asarray(x) #obtain huMoments as an array
        animate(y)	#call function to create a bargraph with the values for HuMoments       
#*******************************************************************************

			
        for yellowMask in yellowMaskList:#Check every yellow mask to see if it is contiguous with blue mask
            res = cv2.bitwise_or(maskList[i][0],yellowMask[0])
            resD = cv2.dilate(res,kernel,iterations =1) 
            contours, hierarchy = cv2.findContours(resD,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #print(len(contours))
            if(len(contours) == 1):
                birdMask = cv2.bitwise_or(birdMask,res)
                contiguousYellows += 1
                
            #If we've added yellows, check for blues that are also contiguous (add head)
            for j in xrange(len(maskList)):
                if(j != i):
                    #res1 = maskList[j][0] + birdMask
                    res1 = cv2.bitwise_or(maskList[j][0],birdMask)
                    #res1 = cv2.morphologyEx(res1,cv2.MORPH_CLOSE,kernel)
                    resD1 = cv2.dilate(res1,kernel,iterations =1) 
                    contours, hierarchy = cv2.findContours(resD1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    #print(len(contours))
                    if(len(contours) == 1):
                        #birdMask = birdMask + res1
                        birdMask = cv2.bitwise_or(birdMask,res1)
        if contiguousYellows >= 2:
			showMe('birdMask', birdMask*255)
			break
 
cap.release()   
cv2.destroyAllWindows()
sys.exit()
