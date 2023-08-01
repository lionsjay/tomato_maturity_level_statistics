import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy
from PIL import Image

# image=cv2.imread('F:\dataset/19th2_tomato\8th\sequoia/0318\GRRI/IMG_230418_072204_0001_GRRI.TIF', 1)
# image=Image.open('F:\dataset/19th2_tomato\8th\sequoia/0318/NIR/IMG_230418_072204_0001_NIR.TIF')
# image=cv2.imread('F:\dataset\crop_images/6th_100/18.jpg')
image=cv2.imread('F:\dataset\crop_images/tomato_try13.PNG')
# image=cv2.imread('F:\dataset/19th_tomato/6th\sequoia/3l_train\images\RGB\IMG_221027_072548_0018_RGB.JPG')
# print(image)
arr = numpy.array(image)
size = image.shape
print(numpy.shape(arr))
print(numpy.max(arr))
print(numpy.min(arr))
# image=cv2.resize(image, (size[1],size[0]))
# image=cv2.resize(image, (1280,960))
# RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# for i in range(size[1]):
#     for j in range(size[0]):
#         if(int(image[j,i][2])-int(image[j,i][1])<-10): #R-G值
#             image[j,i]=[0,0,0]

def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
        print(HSV[y,x])
        print(image[y,x])
        print('r-g='+str(int(image[y,x][2])-int(image[y,x][1])))

# cv2.imshow("imageHSV",RGB)
cv2.imshow('image',HSV)
cv2.setMouseCallback("image",getpos)
cv2.waitKey(0)
