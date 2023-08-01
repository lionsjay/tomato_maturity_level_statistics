import os
from PIL import Image
import cv2
import numpy as np
import math

#圆形头像
#https://blog.csdn.net/u013091013/article/details/103876879

def ellipse(ima): #opencv版
    # path_name = os.path.dirname(img_path)
    # #print(path_name) #F:\dataset
    # cir_file_name = 'cir_img.png'
    # cir_path = path_name + '/' + cir_file_name
    # print(cir_path)
    # ima = Image.open(img_path).convert("RGBA")
    #size = ima.size
    # size = ima.size
    size = ima.size
    rh = ima.shape[1]
    rw = ima.shape[0]
    # print(size)
    # 因为是要圆形，所以需要正方形的图片
    #r2 = min(size[0], size[1])
    # rh = size [1]
    # rw = size [0]
    # if size[0] != size[1]:
    #     ima = ima.resize((r2, r2), Image.ANTIALIAS)
    # 最后生成圆的半径
    #r3 = int(r2/2)
    # imb = Image.new('RGBA', (rw, rh),(255,255,255,0))
    # imb=np.zeros((size [0],size [1]))
    imb=[]
    # pima = ima.load() # 像素的访问对象
    # pimb = imb.load()
    
    rox = float(rw/2) #圆心横坐标
    roy = float(rh/2)
    # print(rox,roy)
    # print(ima[200,200][0])

    if(rw>=rh):
        focal=(pow(rw/2,2)-pow(rh/2,2))**0.5
        #(f1x,f1y),(f2x,f2y)為焦距的座標
        f1x=rox-focal
        f1y=roy
        f2x=rox+focal
        f2y=roy
        for i in range(rw):
            for j in range(rh):
                length=(pow((i-f1x),2)+pow((j-f1y),2))**0.5+(pow((i-f2x),2)+pow((j-f2y),2))**0.5
                threshold=rw
                #print(length)
                if(length>threshold):
                    ima[i,j]=[0,0,0]
                    # ima[i,j]=np.nan
                    
                    
                



    else:
        focal=(pow(rh/2,2)-pow(rw/2,2))**0.5
        #(f1x,f1y),(f2x,f2y)為焦距的座標
        f1x=rox
        f1y=roy-focal
        f2x=rox
        f2y=roy+focal
        for i in range(rw):
            for j in range(rh):
                length=(pow((i-f1x),2)+pow((j-f1y),2))**0.5+(pow((i-f2x),2)+pow((j-f2y),2))**0.5
                threshold=rh
                if(length>threshold):
                    ima[i,j]=[0,0,0]
                    # ima[i,j]=np.nan

    #print(imb[200,200])               
                
    
    # print('threshold='+str(threshold))
    # print('focal='+str(focal))
 
   
 
    #imb.save(cir_path)
    return ima
    #return cir_path
               
                
    
    # print('threshold='+str(threshold))
    # print('focal='+str(focal))
    # print('max='+str(np.nanmax(pimb)))
    # print('min='+str(np.nanmin(pimb)))
    # print('mean='+str(np.nanmean(pimb)))
 
   
 
    

if __name__=='__main__':
    img_path='F:\dataset\crop_images/tomato_try6a.PNG'
    path_name = os.path.dirname(img_path)
    #print(path_name) #F:\dataset
    cir_file_name = 'tomato_try6b.png'
    cir_path = path_name + '/' + cir_file_name
    ima=cv2.imread(img_path)
    cv2.imwrite(cir_path,ellipse(ima))
    size=ima.shape
    area=size[0]*size[1]*0.25*math.pi
    area2=0
    for i in range (size[1]):
        for j in range(size[0]):
            if(ima[j,i][0]>0 and ima[j,i][1]>0 and ima[j,i][2]>0):
                area2=area2+1
    
    print(area,area2)

