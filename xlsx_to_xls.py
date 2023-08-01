# encoding: utf-8
from ctypes import *
import time
import win32com.client as win32
import os
import pandas as pd
import numpy as np

def excel_merge(dir):
    #dir = "F:/dataset/19th_tomato\zz"#設定工作路徑
    #新建列表，存放檔名（可以忽略，但是為了做的過程能心裡有數，先放上）
    
    filename_excel = []
    #新建列表，存放每個檔案資料框（每一個excel讀取後存放在資料框）
    frames = []
    for root,dirs,files in os.walk(dir):
        # print(root)
        for file in files:
            #print(os.path.join(root,file))
            filename_excel.append(os.path.join(root,file))
            df = pd.read_excel(os.path.join(root,file)) #excel轉換成DataFrame
            frames.append(df)
    #列印檔名
    print(filename_excel) 
    #合併所有資料
    result = pd.concat(frames) 
    #檢視合併後的資料
    result.head()
    result.shape
    result.to_csv(dir+'/merged.csv',sep='\t',index = False)#儲存合併的資料到電腦D盤的merge資料夾中，並把合併後的檔案命名為a12.csv
def transform(parent_path,out_path, delete_flag=True):
    fileList = os.listdir(parent_path)  #文件夹下面所有的文件
    num = len(fileList)
    for i in range(num):
        file_Name = os.path.splitext(fileList[i])   #文件和格式分开
        
        if file_Name[1] == '.xlsx':
            transfile1 = parent_path+'/'+fileList[i]  #要转换的excel
            #F:/dataset/19th_tomato/ss/8l_test_4th.xlsx
            transfile2 = out_path+'/'+file_Name[0]    #转换出来excel
            #F:/dataset/19th_tomato/ss/xls/8l_test_4th
            print(transfile1)
            print(transfile2)
            excel=win32.gencache.EnsureDispatch('excel.application')
            excel.Visible = False
            excel.DisplayAlerts = False
            pro=excel.Workbooks.Open(transfile1)   #打开要转换的excel
            #time.sleep(3)
            pro.SaveAs(transfile2+".xls", FileFormat=56)  #另存为xls格式
            pro.Close(SaveChanges=1)
            
            excel.Application.Quit()
            if delete_flag:
                os.remove(transfile1)

if __name__=='__main__':
    path1=r'F:/dataset/19th_tomato/ss'  #待转换文件所在目录
    path2=r'F:/dataset/19th_tomato/ss/xls'  #转换文件存放目录
    #path1=r"E:\untitled1\test_report_demo"  #待转换文件所在目录
    # transform(path1, path2)
    excel_merge(path1)
