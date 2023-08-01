import cv2.cv2 as cv
import os
import gdal
gdal_translate -of GTiff -ot Byte -scale 0 65535 0 255 16bit_img.tif 8bit_img.tif

# path = os.getcwd()  # 获取代码所在目录
path =r'F:/micasense_imageprocessing_sequoia-master\data\Sequoia/test/NDVI'  # 获取代码所在目录
path2 =r'F:/micasense_imageprocessing_sequoia-master\data\Sequoia/test/NDVI_JPG'  # 获取代码所在目录
tif_list = [x for x in os.listdir(path) if x.endswith(".tif")]   # 获取目录中所有tif格式图像列表
for num,i in enumerate(tif_list):      # 遍历列表
    print(path+'/'+i)
    img = cv.imread(path+'/'+i,-1)       #  读取列表中的tif图像
    cv.imwrite(path2+'/'+i.split('.')[0]+".jpg",img)    # tif 格式转 jpg 并按原名称命名
    # print(i.split('.')[0]+".jpg",img)
