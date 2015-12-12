from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
	#print lensig
        symset=list(set(signal))
        numsym=len(symset)
	#print numsym
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
colorIm=Image.open('mask.jpg')
#print colorIm
grayIm=colorIm.convert('L')
colorIm=np.array(colorIm)
grayIm=np.array(grayIm)
N=5
S=grayIm.shape
E=np.array(grayIm)
#print E
for row in range(S[0]):
        for col in range(S[1]):
                Lx=np.max([0,col-N])
		#print Lx
                Ux=np.min([S[1],col+N])
		#print Ux
                Ly=np.max([0,row-N])
                Uy=np.min([S[0],row+N])
                region=grayIm[Ly:Uy,Lx:Ux].flatten()
		#print region
                E[row,col]=entropy(region)
(means, stds) = cv2.meanStdDev(E)
print means
print stds		
plt.subplot(2,4,1)
plt.imshow(colorIm)

plt.subplot(2,4,2)
plt.imshow(grayIm, cmap=plt.cm.gray)

plt.subplot(2,4,3)
plt.hist(E);

plt.subplot(2,4,4)
plt.imshow(E, cmap=plt.cm.jet)
plt.xlabel('Entropy in 10x10 neighbourhood')
plt.colorbar()

plt.show()
