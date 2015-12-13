import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('final.jpg',0)
org_img=cv2.imread('final.jpg')
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
mask = cv2.morphologyEx(th1, cv2.MORPH_OPEN, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
bw=mask
_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
print len(contours)
centres = []
tup_squares=()
tup_roi=()
area=0
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
  roi=org_img[y:y+h,x:x+w]
  #print roi.size
  tup_roi=(roi,)+tup_roi

for k in range(len(tup_roi)):
  print tup_roi[k].size
  if tup_roi[k].size > area:
    area= tup_roi[k].size
    out= tup_roi[k]
    print tup_roi[k].size
#print tup_roi
cv2.imshow('image',out)
cv2.imwrite("mask.jpg",out)
titles = ['Original Image', 'Global Thresholding (v = 127)','masked image']
images = [img, bw,out]

for i in xrange(3):
    plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


