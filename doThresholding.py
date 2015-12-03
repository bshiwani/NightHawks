
import cv2
import numpy as np
from numpy import linalg as LA
import sys
from matplotlib  import pyplot as plt
import colorThreshold_m as ct
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

def actualWork():
	myTh = ct.thresholding()
	myTh.startRecordingThreshold(myImg)  
	cv2.waitKey(0)
	myTh.saveThresholds(fileName)
	cv2.destroyAllWindows()


fileName = str(raw_input("Enter the name of file to save thresholds.\n"))
repeat = True
while (repeat == True):
	cap = cv2.VideoCapture(0)
	print ' press q to stop streaming, and start thresholding\n'
	while cv2.waitKey(1)& 0xFF != ord('q'):
	    ret,myImg = cap.read()   
	    myImg = cv2.medianBlur(myImg,3)
	    cv2.imshow('orig',myImg)
	cap.release()
	cv2.destroyWindow('orig')
	actualWork()	
	toDoAgain = raw_input("Want to exit? press 'q' then enter.\n To continue press any other key then enter\n")
	if toDoAgain == "q":
	    repeat = False
	else:	
	    fileName = str(raw_input("Enter the name of file to save thresholds.\n"))
sys.exit()






