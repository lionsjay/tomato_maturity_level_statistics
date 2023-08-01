import cv2
import numpy as np
import pandas as pd
import math
from ellipse import ellipse

image = cv2.imread('F:\dataset\crop_images/tomato_try.png') # 读取的是BGR  # 參照影象
ellipse(image)
LAB=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
l=LAB[0]
a=LAB[1]
b=LAB[2]
h=HSV[0]
s=HSV[0]
v=HSV[0]
# print(np.max(h),np.max(s),np.max(v))
filtration_capacity=75
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 先轉成灰階
# maxium=np.percentile(img_gray, filtration_capacity, interpolation='midpoint') #https://www.796t.com/content/1550429124.html
# minimum=np.percentile(img_gray,  (100-filtration_capacity)+100*(1-0.25*math.pi), interpolation='midpoint')
# #https://blog.csdn.net/JNingWei/article/details/77747959
# ret, thresh1 = cv2.threshold(img_gray, maxium, 255, 4)  # 利用 threshold 過濾出 #type = 4 ：此时小于阈值的像素点保持原色，大于阈值的像素点置0
# ret, thresh2 = cv2.threshold(img_gray, minimum, 255, 3)  # type = 3 ：此时大于阈值的像素点置填充色，小于阈值的像素点置0
# # concat_pic = np.concatenate([img_gray, thresh], axis=1)
# img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)  
# res = cv2.bitwise_and(image, image,mask=thresh1)
# res = cv2.bitwise_and(res, res,mask=thresh2)

# cv2.imwrite("F:\dataset\crop_images/image.jpg", res)

# img_rg=image[2]-image[3] #g-r
maximum=np.percentile(img_gray, filtration_capacity, interpolation='midpoint') #https://www.796t.com/content/1550429124.html
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 先轉成灰階
ret, thresh1 = cv2.threshold(img_gray, maximum, 255, 4)  # 利用 threshold 過濾出 #type = 4 ：此时小于阈值的像素点保持原色，大于阈值的像素点置0
img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)  
res = cv2.bitwise_and(image, image,mask=thresh1)
image=res
image_rg=image[:,:,1]-image[:,:,2]
rg_maximum=np.percentile(image_rg,  85*0.25*math.pi, interpolation='higher')#下四分位数

print(maximum)
for i in range(image.shape[1]):
    for j in range(image.shape[0]):
        if(image_rg[j,i]>rg_maximum):
            
            image[j,i]=[0,0,0]

cv2.imwrite("F:\dataset\crop_images/image.jpg", image)



# while(True):
#     try:
#         cv2.waitKey(100)
#     except Exception:
#         cv2.destroyWindow("image")
#         break
        
# cv2.waitKey(0)
# cv2.destroyAllWindow()

# cv2.imwrite('mask.jpg', res)
# res.replace(0,np.nan)
sum=0

# def non_zero_mean(np_arr, axis):
#     """ndarray按行/列求非零元素的均值。
#     axis=0按列
#     axis=1按行
#     """
#     exist = (np_arr != 0)
#     num = np_arr.sum(axis=axis)
#     den = exist.sum(axis=axis)
#     return num/den
# # np.where(res,res,np.nan)
# #https://blog.csdn.net/m0_37833297/article/details/113123351
# exist = (res != [0,0,0])
# no_exist = (res == [0,0,0])
# mean_value = res[0].sum()/exist.sum()
# v=non_zero_mean(res, axis=0)
# print(v)
# print(image.shape[1]*image.shape[0])
# # print (mean_value)
# print(np.nanmean(res))




# a = np.array([[0,1],[4,3],[0,0],[9,0],[0,0],[7,0]])
# exist = (a != [0,0])

# mean_value = a[1].sum()/exist.sum()
# v=non_zero_mean(res, axis=0)
# print(v)
# print(exist.sum())
# print (mean_value)
# print(np.nanmean(a))
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower_green = np.array([35, 43, 46])
# upper_green = np.array([77, 255, 255])
# mask = cv2.inRange(hsv, lower_green, upper_green)
# res = cv2.bitwise_and(image, image, mask=mask)
# res2 =  cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imwrite('mask.jpg',res2)
# cv2.imshow('Input', image)
# cv2.imshow('Result', res)
# cv2.waitKey(0)


# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 產生一張灰階的圖片作為遮罩使用
# ret, mask1  = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)  # 使用二值化的方法，產生黑白遮罩圖片
# logo = cv2.bitwise_and(img, img, mask = mask1 )  # logo 套用遮罩

# bg = cv2.imread('meme.jpg')                      # 讀取底圖
# ret, mask2  = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)      # 使用二值化的方法，產生黑白遮罩圖片
# bg = cv2.bitwise_and(bg, bg, mask = mask2 )      # 底圖套用遮罩

# output = cv2.add(bg, logo)                       # 使用 add 方法將底圖和 logo 合併
# cv2.imshow('oxxostudio', output)