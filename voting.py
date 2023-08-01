import cv2
import os
import numpy as np
import openpyxl
import numpy
from matplotlib import pyplot as plt
from tqdm import tqdm
from ellipse import ellipse
from excel_clear import predict_excel_clear
from PIL import Image
from openpyxl import load_workbook
import math
import sys   
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def voting(excel_path):
    excel_path2=excel_path[:-5]+'2.xlsx'

    test_data = pd.read_excel(excel_path)
    test_data.sort_values(by='id',inplace=True,ascending=True) #ascending 默认等于True，按从小到大排列，改为False 按从大到小排
    pd.DataFrame(test_data).to_excel(excel_path2,header=True,index=False)
    workbook = openpyxl.load_workbook(excel_path2)
    sheet = workbook.worksheets[0]
    # test_data.to_excel(excel_path2, sheet_name='sheet1', index=False)
    # workbook.save(excel_path2)

    # print(test_data[120:140])
    predict_class = test_data['ripness'].values
    ids = test_data['id'].values
    out_ripness = []
    out_ids = []
    out_ids = ids[0]
    i = 0
    j = 2
    tomato_number = 0
    m = [0,0,0,0,0]
    # print('id  ripness')
    while i < len(ids) - 1:
        if ids[i] == ids[i+1]:
            out_ripness.append(abs(predict_class[i]))
            
        else:
            #求眾數 #https://blog.csdn.net/u013066730/article/details/108844068
            
            counts = np.bincount(out_ripness)
            if counts.size :       #https://stackoverflow.com/questions/46344772/valueerror-attempt-to-get-argmax-of-an-empty-sequence
                large = np.argmax(counts)
            # print(i,counts.size)
            
            tomato_number = tomato_number + 1
            m[large-1]=m[large-1]+1

            for column in range(j,i):
                # if(math.isnan(int(sheet['A'+str(column+2)]))==False):                
                    sheet['S'+str(column+2)] = 0 - large
                    column=column+1
                
            # print(str(ids[i])+'   '+str(0-large))


            out_ripness.clear()
            out_ids = ids[i+1]
            j=i+1

        i=i+1
            
    workbook.save(excel_path2)
    print('tomato_number='+str(tomato_number))
    print('[m1,m2,m3,m4,m5]='+str(m))

    test_data = pd.read_excel(excel_path2)
    test_data.sort_values(by='frame',inplace=True,ascending=True) #ascending 默认等于True，按从小到大排列，改为False 按从大到小排
    pd.DataFrame(test_data).to_excel(excel_path,header=True,index=False)


def classification_voting(predicted,ids):
    # ids.append(0)
    # predicted.append(0)
    predicted_class=np.stack((ids, predicted), axis=1)
    
    
    out_ripness = []
    i = 0
    j = 0
    predicted_after_voting=[]
    while i < len(ids) - 1:
        
        # if(i==0):
        #     out_ripness.append(abs(predicted_class[0][1]))
        if ids[i] == ids[i+1]:
            out_ripness.append(abs(predicted_class[i][1]))
            
        else:
            out_ripness.append(abs(predicted_class[i][1]))
            # print(j,i)
            #求眾數 #https://blog.csdn.net/u013066730/article/details/108844068
            
            counts = np.bincount(out_ripness)
            if counts.size :       #https://stackoverflow.com/questions/46344772/valueerror-attempt-to-get-argmax-of-an-empty-sequence
                large = np.argmax(counts)
            # print(i,counts.size)
            
            
            
            

            for column in range(j,i+1):
                # if(math.isnan(int(sheet['A'+str(column+2)]))==False):                
                    # predicted_after_voting[column] = 0 - large
                    predicted_after_voting.append( 0 - large)
                    # column=column+1
                
            # print(str(ids[i])+'   '+str(0-large))


            # predicted_after_voting.append(out_ripness)
            out_ripness.clear()
            out_ids = ids[i+1]
            j=i+1

        i=i+1

        if(i == len(ids) - 2):
            # print(j,i)
            #求眾數 #https://blog.csdn.net/u013066730/article/details/108844068
            out_ripness.append(abs(predicted_class[i][1]))
            counts = np.bincount(out_ripness)
            if counts.size :       #https://stackoverflow.com/questions/46344772/valueerror-attempt-to-get-argmax-of-an-empty-sequence
                large = np.argmax(counts)
            # print(i,counts.size)
            
            
            
            

            for column in range(j,i+2):
                # if(math.isnan(int(sheet['A'+str(column+2)]))==False):                
                    # predicted_after_voting[column] = 0 - large
                    predicted_after_voting.append( 0 - large)
                    # column=column+1
                
            # print(str(ids[i])+'   '+str(0-large))


            # predicted_after_voting.append(out_ripness)
            out_ripness.clear()
            out_ids = ids[i+1]
            j=i+1

    predicted_after_voting = np.array(predicted_after_voting)

    # print(j,i)
    return predicted_after_voting
# print(test_data[0:10])

if __name__=='__main__':
    excel_path='F:\dataset/lin_tomato\sequoia_mot/5th/predict.xlsx'
    voting(excel_path)
    # excel_path2='F:\dataset/19th_tomato\sequoia_mot/6th/predict2.xlsx'
