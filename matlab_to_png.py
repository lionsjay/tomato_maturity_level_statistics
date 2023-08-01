import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
#https://codeantenna.com/a/s6o8jWOmgH

data_dir = 'F:\dataset/GRE'
path = 'F:\dataset/NDVI2/'
# data_dir = './CoNSeP/Train/Labels'
# path = './CoNSeP/Train/Labels/'
img_ids    =   sorted(os.listdir(data_dir))
# print(len(img_ids))
# print(img_ids[2][-4:])
# input_mat.keys()

for img_id in img_ids:
    print(img_id)
    if(img_id[-4:]=='.mat'):
        dataFile =  data_dir +'/'+ img_id  # 单个的mat文件
        data = scio.loadmat(dataFile)
        
        print(dataFile)
        print(type(data))
        # print (data['data'])
        # 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
        a=data['GRE','RED']
        # a=data['inst_map']
        # 取出需要的数据矩阵

        # 数据矩阵转图片的函数
        def MatrixToImage(data):
            data = data*255
            new_im = Image.fromarray(data.astype(np.uint8))
            return new_im

        new_im = MatrixToImage(a)
        # plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
        # new_im.show()
        new_im.save(path+img_id[:-4] + '.png') # 保存图片
