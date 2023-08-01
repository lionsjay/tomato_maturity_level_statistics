# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:58:54 2022

@author: lionsjay
"""


#標註完資料後，把id跟前一個image sequences做對齊

#itzu_tomato
#img8213=273
#video1=274-504
#video4=505-697

#19th_tomato
#2nd-3l=1-86
#2nd-5l=87-133
#3rd-3l=134-224
#3rd-5l=225-319
#4th-3l=320-383
#4th-5l=384-455
#5th-3l=456-547
#5th-5l=548-637
#6th-3l=638-684
#6th-5l=685-764
#7th-3l=765-812
#7th-5l=813-849

#lin_tomato(2l_train,4l_train)
#3rd=0-32,33-59
#4th=60-93,94-129
#5th=130-154,155-178
#6th=179-198,199-206
#7th=207-226,227-237
#8th=238-249,250-254
#lin_tomato(5l_test)
#3rd=0-30
#4th=31-67
#5th=68-87
#6th=88-107
#7th=108-125
#8th=126-140
import os
# cuhksysu_path='C:/anydesk/CUHK-SYSU/CUHK-SYSU'
# sysu_path='C:/anydesk/CUHK-SYSU/SYSU'
cuhksysu_path='F:\dataset\lin_tomato/8th\sequoia/4l_train\labels_with_ids' #原來的
sysu_path='F:\dataset\lin_tomato/8th\sequoia/4l_train\labels_with_ids2'  #後來的
filelist=os.listdir(cuhksysu_path)

if not os.path.isdir(sysu_path):
    os.mkdir(sysu_path)
print(sysu_path)
maxium=0
start_ids=250
for i in  range (len(filelist)):
    isExist = os.path.exists(sysu_path+'/'+filelist[i])
    if  isExist:
        os.remove(sysu_path+'/'+filelist[i])
    with open(cuhksysu_path+'/'+filelist[i]) as f:
     #print(filelist[i])   
     for line in f.readlines():
        s = line.split(' ')
        if(int(s[1])>maxium):
            maxium=int(s[1])
        ids=int(s[1])
        ncx=s[2]
        ncy=s[3]
        nw=s[4]
        nh=s[5]
        if(ids==-1):
            ids=-1
        else:
            ids=(ids+1)+start_ids #274
        #if (os.path.exists(sysu_path+'/'+filelist[i])==False):
         #os.mkdir(sysu_path+'/'+filelist[i])
        g = open(sysu_path+'/'+filelist[i],'a+')        
        g.write('0'+' '+str(ids)+' '+str(ncx)+' '+str(ncy)+' '+str(nw)+' '+str(nh))
        g.close()

#補齊沒有id的txt檔
for i in range(len(filelist)):
    isExist = os.path.exists(sysu_path+'/'+filelist[i])
    if not isExist:
        f=open(sysu_path+'/'+filelist[i],'w+')
        # print('create')
        f.close()
print('start_id='+str(start_ids))
print('end_id='+str(start_ids+maxium))
# #補齊沒有id的txt檔
# for i in range(len(filelist)):
#     isExist = os.path.exists(sysu_path+str(i).rjust(8,'0')+'.txt')
#     if not isExist:
#         f=open(sysu_path+'/'+str(i).rjust(8,'0')+'.txt','w+')
#         # print('create')
#         f.close()