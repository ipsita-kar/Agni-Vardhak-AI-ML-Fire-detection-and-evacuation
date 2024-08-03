#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2;
import math;
import numpy as np;
from guided_filter_tf.guided_filter import guided_filter
def DarkChannel(im,sz):
b,g,r = cv2.split(im)
dc = cv2.min(cv2.min(r,g),b);
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
dark = cv2.erode (dc, kernel)
return dark
def AtmLight(im,dark):
[h,w] = im.shape[:2]
imsz = h*w
numpx = int(max(math.floor(imsz/1000),1))
darkvec = dark.reshape(imsz);
imvec = im.reshape(imsz,3);
indices = darkvec.argsort();
indices = indices [imsz-numpx::]
atmsum = np.zeros([1,3])
for ind in range(1, numpx):
atmsum = atmsum + imvec[indices [ind]]
A = atmsum / numpx;
return A
def TransmissionEstimate(im, A, sz):
omega = 0.95;
im3 = np.empty(im.shape,im.dtype);
for ind in range(0,3):
im3[:,:,ind] = im[:,:,ind]/A[0,ind]
transmission = 1 - omega*DarkChannel (im3, sz);
return transmission
def Guidedfilter (im, p, r, eps):
mean_I = cv2.boxFilter(im,cv2.CV_64F, (r,r));
mean_p = cv2.boxFilter(p, cv2.CV_64F, (r,r));
mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F, (r,r));
cov_Ip = mean_Ip - mean_I*mean_p;
mean_II = cv2.boxFilter(im*im,cv2.CV_64F, (r,r));
var_I = mean_II - mean_I*mean_I;
a = cov_Ip/(var_I + eps);
b = mean_p - a*mean_I;
mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
mean_b = cv2.boxFilter(b,cv2.CV_64F, (r,r));
q = mean_a*im + mean_b;
return q;
def TransmissionRefine(im,et):
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
gray = np.float64(gray)/255;
r = 60;
eps = 0.0001;
t = Guidedfilter(gray,et,r,eps);
return t;
def Recover (im,t, A, tx = 0.1):
res = np.empty(im.shape, im.dtype);
t = cv2.max(t,tx);
for ind in range(0,3):
res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0, ind]
return res
#
#
# def Video (path):
#
image = cv2.VideoCapture(path)
while True:
done, img = image.read()
#
I = img.astype('float64')/255
#
dark = DarkChannel (I,15)
#
A = AtmLight(I, dark)
#
te = TransmissionEstimate (I, A, 15)
#
t = TransmissionRefine (img,te)
#
J = Recover (I,t,A,0.1)
#
J = cv2.resize(J, (0, 0), fx = 0.5, fy = 0.5)
#
cv2.imshow('frame',J)
#
cv2.imwrite("./J.me.mp4", J*255)
#
if cv2.waitKey(1) & 0xFF == ord('q'):
#
break
# if __name__ == '_main__':
#
#
import sys
Video(r"C:\Users\Lohesh\Downloads\ominous-hazy-atmosphere-shorts-1280-ytshorts.savetube.me.mp4")
def Video(input_path, output_path):
image = cv2.VideoCapture(input_path)
if not image.isopened():
print("Error: Unable to open the input video.")
return
frame_width = int(image.get(3))
frame_height = int(image.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
if not out.isOpened():
print("Error: Unable to create the output video.")
return
while True:
done, img = image.read()
if not done:
break
I = img.astype('float64') / 255
I = cv2.bilateralFilter (I, d=9, sigmaColor=75, sigmaSpace=75)
dark = DarkChannel (I, 15)
A = AtmLight(I, dark)
te = TransmissionEstimate (I, A, 15)
t = TransmissionRefine (img, te)
J = Recover(I, t, A, 0.1)
cv2.imshow('frame', J)
J_uint8 = (J * 255).astype('uint8')
out.write(J_uint8)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
out.release()
cv2.destroyAllWindows()
if __name__ == '_main__':
Video(r"C:\Users\Lohesh\Downloads\Notes\smh_Hack\dark_channel\ominous-hazy-atmosphere-shorts-1280-ytshorts.savetube.me.mp4", r"C:\Users\Lohesh\Downloads\Notes\smh_Hack\dark_channel\output.me.mp4")
```

