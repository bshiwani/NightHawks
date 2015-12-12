from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

def entropy(signal):
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

img = cv2.imread('3.jpg',0)
org_img=cv2.imread('3.jpg')
#img = cv2.medianBlur(img,5)
#cv2.imshow('Original Image',org_img)
cv2.waitKey(0)

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
  tup_roi=(roi,)+tup_roi

for k in range(len(tup_roi)):
  if tup_roi[k].size > area:
    area= tup_roi[k].size
    out= tup_roi[k]
cv2.imwrite("mask.jpg",out)
colorIm=Image.open('mask.jpg')
grayIm=colorIm.convert('L')
gray_img=grayIm
colorIm=np.array(colorIm)
grayIm=np.array(grayIm)
N=5
S=grayIm.shape
E=np.array(grayIm)
for row in range(S[0]):
        for col in range(S[1]):
                Lx=np.max([0,col-N])
		Ux=np.min([S[1],col+N])
                Ly=np.max([0,row-N])
                Uy=np.min([S[0],row+N])
                region=grayIm[Ly:Uy,Lx:Ux].flatten()
                E[row,col]=entropy(region)


(means, stds) = cv2.meanStdDev(E)
hist,bins = np.histogram(E,300,[0,6])
hist1,bins1 = np.histogram(E,300,[4.75,6])


#counts, bins, bars = plt.hist(E)
#print hist
#print bins
average = sum(hist)/len(bins)
print average
print "next iteration"
#print hist1
average = sum(hist1)/len(bins1)
print average
#print bins1
print "full image mean"
#print bars
print means
print stds

plt.subplot(2,3,1)
plt.imshow(org_img)
plt.xlabel('Original Image')
plt.xticks([]),plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(bw)
plt.xlabel('Global Thresholding (v = 127)')
plt.xticks([]),plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(colorIm)
plt.xlabel('Mask to Calculate Entropy')

plt.subplot(2,3,4)
plt.plot(hist1)
'''
plt.subplot(2,3,4)
plt.imshow(gray_img, cmap=plt.cm.gray)
plt.xlabel('Gray Scale Image')
'''
plt.subplot(2,3,5)
plt.imshow(E, cmap=plt.cm.jet)
plt.xlabel('Entropy in 10x10 neighbourhood')
plt.colorbar()

plt.subplot(2,3,6)
plt.plot(hist);
plt.xlabel('Entropy Histogram')

plt.show()
