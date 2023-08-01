import os
import cv2
import numpy as np
from tqdm import tqdm

# path='F:\dataset\lin_tomato/add\sequoia/0161/images/RGB'
path='F:\dataset/19th2_tomato/4th/sequoia/0247\images\RGB'
filelist=os.listdir(path)
print('翻轉目標:'+path)
pbar = tqdm(total=len(filelist),desc='翻轉照片')
for i in  range (len(filelist)):
    # print({},filelist[i])
    img = cv2.imread(path+'/'+filelist[i])
    output_ROTATE_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(path+'/'+filelist[i], output_ROTATE_180)
    pbar.update()

pbar.close()














