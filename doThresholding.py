
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

#**********Main Code Starts Here************************************
fileName = str(raw_input("Enter the filename to save thresholds to:\n"))
repeat = True
while (repeat == True):
    cap = cv2.VideoCapture(0)
    print('Place object of interest in view and press q to start thresholding\n')
    #Show video feed until user signals to threshold by typing q
    while cv2.waitKey(1)& 0xFF != ord('q'):
        ret,myImg = cap.read()   
        myImg = cv2.medianBlur(myImg,3)
        cv2.imshow('orig',myImg)
    #Stop video feed and close windows
    cap.release()
    cv2.destroyAllWindows()
    #Record threshold values
    actualWork()	
    toDoAgain = raw_input("Press q to exit")
    if toDoAgain == "q":
        repeat = False
    else:	
        fileName = str(raw_input("Enter the name of file to save thresholds.\n"))

sys.exit()



