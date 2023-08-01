# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:03:32 2022

@author: lionsjay
"""

import os

mot_path='C:/Users/lionsjay/Desktop/itzu_tomato_dataset/test/forMOT/gt/gt.txt'
#1,7,279,782,113,116,1,-1,-1,-1
cuhksysu_path='C:/Users/lionsjay/Desktop/itzu_tomato_dataset/test/forMOT/CUHK-SYSU/'
#0 2 0.371296 0.448698 0.0574074 0.0401042
f = open(mot_path,'r')

width=1080
height=1920

lines = len(f.readlines())
#print(lines)

with open(mot_path) as f:
    for line in f.readlines():
        s = line.split(',')
        #1,7,279,782,113,116,1,-1,-1,-1
        #mot_data_fmt: [fn, id, x1, y1, w, h, c=1, c=-1, c=-1, c=-1]
        ids=s[1]
        x1=int(s[2])
        y1=int(s[3])
        w=int(s[4])
        h=int(s[5])
        ncx=(x1+0.5*w)/width
        ncy=(y1+0.5*h)/height
        nw=w/width
        nh=h/height
        print('0'+' '+ids+' '+str(ncx)+' '+str(ncy)+' '+str(nw)+' '+str(nh))
        g = open(cuhksysu_path+ '/'+str(s[0]).rjust(6,'0')+'.txt','a')
        g.write('0'+' '+ids+' '+str(ncx)+' '+str(ncy)+' '+str(nw)+' '+str(nh)+'\n')
        g.close()
        #0 2 0.371296 0.448698 0.0574074 0.0401042
        #cuhk-sysu_data_fmt: [c=0, id, ncx, ncy, nw, nh]
        #print(s[2])