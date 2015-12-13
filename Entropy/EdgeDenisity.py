from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
'''
#Read the image

img = cv2.imread('old.jpg',0)
org_img=cv2.imread('old.jpg')
'''
cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, img1 = cap.read()
	cv2.imshow('img',img1)
	#Thresholding and filling noise for white color
	img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	print th1.shape
	se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
	#se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
	mask = cv2.morphologyEx(th1, cv2.MORPH_OPEN, se1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)
	bw=mask
	cv2.imshow('mask',mask)
	#Finding contours
	_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
	#print len(contours)
	centres = []
	tup_squares=()
	tup_roi=()
	area=0

	#Mark Squares over all contours
	for i in range(len(contours)):
	  if cv2.contourArea(contours[i]) < 2000:
	    continue
	  moments = cv2.moments(contours[i])
	  centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
	  cv2.circle(mask, centres[-1], 3, (0, 0, 0), -1)
	  x,y,w,h = cv2.boundingRect(contours[i])
	  tup_sq=(x,y,w,h)
	  tup_squares=(tup_sq,)+tup_squares
	  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
	  roi=img1[y:y+h,x:x+w]
	  tup_roi=(roi,)+tup_roi

	#Find the biggest contour
	for k in range(len(tup_roi)):
	  if tup_roi[k].size > area:
	    area= tup_roi[k].size
	    out= tup_roi[k]
	cv2.imshow('out',out)
	#Run canny edge detector 
	edge=cv2.Canny(out,50,150)
	cv2.imshow('edge',edge)
	E=np.array(edge)

	#Find the histogram of canny
	hist,bins = np.histogram(edge,255,[250,255])

	#Average the high frequency values or the edge density (250,255)
	average = sum(hist)/len(bins)

	print average
	print "  "
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
