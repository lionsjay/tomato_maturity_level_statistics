#https://www.gushiciku.cn/pl/pjsA/zh-tw
import numpy as np
import cv2
import os
from tqdm import tqdm

#讀取一張圖片
piture_path='F:\dataset/19th_tomato\sequoia_mot/6th\img1'
txt_path='F:\dataset/19th_tomato\sequoia_mot/6th/tracks/img1.txt'
excel_path=''

filelist=os.listdir(piture_path)
size = (4608,3456)
print(size)
#完成寫入物件的建立，第一個引數是合成之後的影片的名稱，第二個引數是可以使用的編碼器，第三個引數是幀率即每秒鐘展示多少張圖片，第四個引數是圖片大小資訊
videowrite = cv2.VideoWriter(r'F:\dataset/19th_tomato\sequoia_mot/6th/test.mp4',-1,2,size)
#videowrite = cv2.VideoWriter(r'F:\test.mp4',-1,20,size)#20是幀數，size是圖片尺寸
# img_array=[]
# for filename in [r'F:/Picture/{0}.jpg'.format(i) for i in range(len(filelist))]:
# for filename in [piture_path+'{0}.jpg'.format(str(i).rjust(8,'0')) for i in range(len(filelist))]:
#     img = cv2.imread(filename)
#     if img is None:
#         print(filename + " is error!")
#         continue
#     img_array.append(img)
pbar = tqdm(total=len(filelist),desc='合成影片')
for i in range(len(filelist)):
    img=cv2.imread(piture_path+'/'+filelist[i])  
    f=open(txt_path,'r')
    for line in f.readlines():
        s = line.split(',')
        s = list(map(int, s))#string轉int
        if(int(s[0])==i):
            # print(s[0],i)
            #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
            #https://blog.gtwang.org/programming/opencv-drawing-functions-tutorial/
            cv2.rectangle(img, (s[2], s[3]), (s[2]+s[4], s[3]+s[5]), (0, 255, 0), 10)
    videowrite.write(img)
    pbar.update()
pbar.close()
print('end!')
