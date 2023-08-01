import cv2
import numpy as np
import matplotlib.pyplot as plt
from ellipse import ellipse 
import math
#读入的图像是BGR空间图像
frame = cv2.imread("F:\dataset\crop_images/tomato_try3.PNG")
# frame = cv2.imread("F:\dataset\crop_images/6th_test/98.jpg")
size = frame.shape 
# 部分1：将BGR空间的图片转换到HSV空间
frame=ellipse(frame)#切成橢圓 
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#部分2：
# 在HSV空间中定义葉子部分
lower_leaf = np.array([15, 0, 0])
upper_leaf = np.array([180, 250, 190]) #要濾掉的
# 在HSV空间中定义绿色
lower_reflect = np.array([0, 0, 252])
upper_reflect = np.array([180, 255, 255])  #要濾掉的
# # 在HSV空间中定义红色
# lower_red = np.array([0, 0, 160])
# upper_red = np.array([15, 255, 245])
 
#部分3：
# 从HSV图像中截取出蓝色、绿色、红色，即获得相应的掩膜
# cv2.inRange()函数是设置阈值去除背景部分，得到想要的区域
leaf = cv2.inRange(hsv, lower_leaf, upper_leaf)
reflect = cv2.inRange(hsv, lower_reflect, upper_reflect)
remained1 = cv2.bitwise_not(leaf)#逐位元not邏輯運算1
remained2 = cv2.bitwise_not(reflect)#逐位元not邏輯運算1
# green_mask = cv2.inRange(hsv, lower_green, upper_green)
# red_mask = cv2.inRange(hsv, lower_red, upper_red)
 
#部分4：
# 将原图像和mask(掩膜)进行按位与
remain=cv2.bitwise_and(frame,frame,mask=remained1)
remain=cv2.bitwise_and(remain,remain,mask=remained2)
# blue_res = cv2.bitwise_and(frame, frame, mask = minus)
# green_res = cv2.bitwise_and(frame, frame, mask = green_mask)
# red_res = cv2.bitwise_and(frame, frame, mask = red_mask)
 
#最后得到要分离出的颜色图像
# res = blue_res + green_res + red_res
 
 
#部分5:将BGR空间下的图片转换成RGB空间下的图片
frame = frame[:,:,::-1]
# blue_res = blue_res[:,:,::-1]
# green_res = green_res[:,:,::-1]
# red_res = red_res[:,:,::-1]
# res = res[:,:,::-1]
# remain = remain[:,:,::-1]
#minus = minus[:,:,::-1]


#部分6：显示图像
# plt.figure(figsize=(14,12))
# plt.subplot(2,2,1),plt.title('original_image'), plt.imshow(frame)
# plt.subplot(2,2,2), plt.imshow(blue_mask, cmap = 'gray')
# plt.subplot(2,2,3), plt.imshow(green_mask, cmap= 'gray')
# plt.subplot(2,2,4), plt.imshow(red_mask, cmap= 'gray')
 
plt.figure(figsize=(14,12))
plt.subplot(2,2,1),plt.title('original_image'), plt.imshow(frame)
plt.subplot(2,2,2), plt.title('leaf'),plt.imshow(leaf)
plt.subplot(2,2,3),plt.title('remain'), plt.imshow(remain)

# plt.subplot(2,2,3),plt.title('red_image'), plt.imshow(red_res)
# plt.subplot(2,2,4), plt.imshow(res)
plt.show()
#左上、右上、左下、右下
# ————————————————
# 版权声明：本文为CSDN博主「鬼 | 刀」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/gaoyu1253401563/article/details/85253511

remain = cv2.cvtColor(remain, cv2.COLOR_BGR2RGB)
area=0
red=0
green=0
blue=0
# print(remain[(size[1]/2),(size[0]/2)])
print(remain[92,19])
for i in range(size[1]):
    for j in range(size[0]):
        
        #OpenCV uses H: 0-179, S: 0-255, V: 0-255
        # hue=hue+HSV[j,i][0]/180
        # saturation=saturation+HSV[j,i][1]/256
        # brightness=brightness+HSV[j,i][2]/256
        if(remain[j,i][0]>4 and remain[j,i][1]>4 and remain[j,i][2]>4):
            area=area+1
            red=red+remain[j,i][0]
            green=green+remain[j,i][1]
            blue=blue+remain[j,i][2]


print('area='+str(area))
print(red/area,green/area,blue/area)
area2=size[0]*size[1]*0.25*math.pi 
print('area2='+str(area2))
# print(red/area2,green/area2,blue/area2)


