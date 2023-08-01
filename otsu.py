
# https://blog.csdn.net/u010128736/article/details/52801310
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ellipse import ellipse 
import math


def yuv_otsu(image):
#image = cv2.imread("F:\dataset\crop_images/tomato_try13.PNG")



    Correction_coefficient=0.9 #校正係數
                    

    # img[y:y+h, x:x+w]
    image = image[int(image.shape[0]*(1-Correction_coefficient)/2):int(image.shape[0]*(1-Correction_coefficient)/2)+int(image.shape[0]*Correction_coefficient), 
                    int(image.shape[1]*(1-Correction_coefficient)/2):int(image.shape[1]*(1-Correction_coefficient)/2)+int(image.shape[1]*Correction_coefficient)]

    image = ellipse(image)
    
    cv2.imwrite("F:\dataset\crop_images/tomato_try17a.png",image)
    h, w = image.shape[:2]

    single_channel_img = np.zeros((h,w,3), dtype='uint8') 
    # print('area='+str(h*w))
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # cv2.imshow('Green', hsv_image)
    single_channel_img=yuv_image[:,:,2]  # cv2 uses `BGR` instead of `RGB` 1:綠色 2:紅色
    # single_channel_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # single_channel_img=image[:,:,2]  # cv2 uses `BGR` instead of `RGB` 1:綠色 2:紅色
    # single_channel_img=image[:,:,2]-image[:,:,1]  # cv2 uses `BGR` instead of `RGB` 1:綠色 2:紅色
    # cv2.imshow('Green', green_img)
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    ret1, th1 = cv2.threshold(single_channel_img, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
    remain=cv2.bitwise_and(image,image,mask=th1)
    # ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
    th1=cv2.cvtColor(th1, cv2.COLOR_BGR2RGB)
    # RGB=cv2.cvtColor(remain, cv2.COLOR_BGR2RGB)
    
    

    

    #面積&像素平均值確認
    area=0
    red=0
    green=0
    blue=0
    for i in range(w):
        for j in range(h):

            if(remain[j,i][0]!=0 and remain[j,i][1]!=0 and remain[j,i][2]!=0):
                area=area+1
                red=red+remain[j,i][0]
                green=green+remain[j,i][1]
                blue=blue+remain[j,i][2]
    # print(red/area,green/area,blue/area)
    # print(w,h,w*h)
    # print('area='+str(area))
    # area2=w*h*0.25*math.pi 
    # print('area2='+str(area2))

    # plt.subplot(131), plt.imshow(image, "gray")
    # plt.title("source image"), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.hist(single_channel_img.ravel(), 256),plt.vlines(ret1,0,180,color="red")
    
    # plt.title("Histogram"), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(remain, "gray")
    # plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
    # plt.show()
    # print(ret1)
    return remain


if __name__=='__main__':
    image = cv2.imread("F:\dataset\crop_images/tomato_try17.PNG")
    image=yuv_otsu(image)
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("F:\dataset\crop_images/tomato_try17b.png",image)
    
