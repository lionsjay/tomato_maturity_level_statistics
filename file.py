# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:35:25 2022

@author: lionsjay
"""
import os
import shutil
import gdal
import numpy as np
import cv2
from warp import warpping
import argparse
np.seterr(divide='ignore',invalid='ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# https://docs.python.org/zh-tw/3/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='lin_tomato', help='番茄資料集名稱')
parser.add_argument('--number', type=str, default='6th', help='第幾次拍攝')
parser.add_argument('--lane', type=str, default='2l_train', help='第幾排番茄')
args = parser.parse_args()

path=os.path.join('F:\dataset',args.datasets,args.number,'sequoia',args.lane)
# path='F:/dataset/19th2_tomato/practice3/0193'
# path='F:\sequoia_calibration/0165'
# path='F:\warpping_try'
# path='F:\dataset\lin_tomato/add\sequoia/0161'
print(path)
print('建立資料夾')
filetype=['NIR','GRE','RED','REG','NDVI','NDRE','GRRI','GNDVI','GRVI','NIR2','GRE2','RED2','REG2','labels_with_ids','labels_with_ids2']
for i in range(len(filetype)):
    if not os.path.isdir(path+'/'+filetype[i]):
        os.mkdir(path+'/'+filetype[i])
filelist=os.listdir(path)
if not os.path.isdir(path+'/images/RGB'):
        os.makedirs(path+'/images/RGB')
if not os.path.isdir(path+'/labels/RGB'):
        os.makedirs(path+'/labels/RGB')

print('移動並分類檔案')
for i in  range (len(filelist)):
    #print(path+'/'+filelist[i])
    #print(filelist[i][-3:])
    if(filelist[i][-7:-4]=='RGB'):
        shutil.copy(path+'/'+filelist[i],path+'/images/RGB')

        #翻轉照片
        rgb_img=cv2.imread(path+'/images/RGB/'+filelist[i])
        output_ROTATE_180 = cv2.rotate(rgb_img, cv2.ROTATE_180)
        cv2.imwrite(path+'/images/RGB/'+filelist[i],output_ROTATE_180) 

    elif(filelist[i][-7:-4]=='NIR'):
        shutil.copy(path+'/'+filelist[i],path+'/'+'NIR2')
    elif(filelist[i][-7:-4]=='GRE'):
        shutil.copy(path+'/'+filelist[i],path+'/'+'GRE2')
    elif(filelist[i][-7:-4]=='RED'):
        shutil.copy(path+'/'+filelist[i],path+'/'+'RED2')
    elif(filelist[i][-7:-4]=='REG'):
        shutil.copy(path+'/'+filelist[i],path+'/'+'REG2')
#shutil.copy(src.des)

    if((filelist[i][-4:]=='.TIF')|(filelist[i][-4:]=='.tif')|(filelist[i][-7:]=='RGB.JPG')|(filelist[i][-7:]=='RGB.jpg')):
        #print(filelist[i][-3:])
        #print(path+'/'+filelist[i])
        os.remove(path+'/'+filelist[i])

red_filelist=os.listdir(path+'/RED2')
nir_filelist=os.listdir(path+'/NIR2')
reg_filelist=os.listdir(path+'/REG2')
gre_filelist=os.listdir(path+'/GRE2')

print('圖像畸變校正')

# print('NIR')
mtx=np.array([[1.08663852e+03 ,0.00000000e+00 ,6.29666558e+02],
 [0.00000000e+00 ,1.08488939e+03 ,4.50833261e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])
dist=np.array( [[-0.4256501   ,0.3791636   ,0.00241498  ,0.00257393 ,-0.42547737]])
for i in  range (len(nir_filelist)):
    
    img = cv2.imread(path+'/NIR2/'+nir_filelist[i])
    # img = cv2.imread('fisheye_sample.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    img_undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    cv2.imwrite(path+'/NIR/'+nir_filelist[i], img_undistorted)

# print('RED')
mtx=np.array([[1.08842144e+03 ,0.00000000e+00 ,6.47072931e+02],
 [0.00000000e+00 ,1.08641657e+03 ,4.81081155e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])
dist=np.array([[-0.4099875   ,0.24144281 , 0.00466022 , 0.00229793 ,-0.01587393]])
for i in  range (len(red_filelist)):
   
    img = cv2.imread(path+'/RED2/'+red_filelist[i])
    
    # img = cv2.imread('fisheye_sample.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))  #alpha=0，视场（两张纸变大了发现没有？）会放大，alpha=1，视场不变
    img_undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    cv2.imwrite(path+'/RED/'+red_filelist[i], img_undistorted)

# print('REG')
mtx=np.array([[1.08898951e+03 ,0.00000000e+00 ,6.51466746e+02],
 [0.00000000e+00 ,1.08662943e+03 ,4.43400190e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])
dist=np.array([[-0.42175293,  0.32584872,  0.00565194,  0.00261993, -0.27278407]])
for i in  range (len(reg_filelist)):
    
    img = cv2.imread(path+'/REG2/'+reg_filelist[i])
    # img = cv2.imread('fisheye_sample.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    img_undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    cv2.imwrite(path+'/REG/'+reg_filelist[i], img_undistorted)

# print('GRE')
mtx=np.array([[1.08617369e+03, 0.00000000e+00, 6.72378687e+02],
 [0.00000000e+00, 1.08384896e+03, 4.72553508e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist=np.array([[-0.41009727,  0.25376652,  0.00568954,  0.00229671, -0.06988319]])
for i in  range (len(gre_filelist)):
    
    img = cv2.imread(path+'/GRE2/'+gre_filelist[i])
    # img = cv2.imread('fisheye_sample.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    img_undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    cv2.imwrite(path+'/GRE/'+gre_filelist[i], img_undistorted)

print('圖像扭轉(warpping)')

warpping(path+'/RED',path+'/RED')
warpping(path+'/NIR',path+'/NIR')
warpping(path+'/REG',path+'/REG')
warpping(path+'/GRE',path+'/GRE')

red_filelist=os.listdir(path+'/RED')
nir_filelist=os.listdir(path+'/NIR')
reg_filelist=os.listdir(path+'/REG')
gre_filelist=os.listdir(path+'/GRE')

print('生產NDVI影像')
# red=path+'/RED/'+red_filelist[i]   
# nir=path+'/NIR/'+nir_filelist[i]
# reg=path+'/REG/'+reg_filelist[i]
for i in  range (len(red_filelist)):
    #print(red_filelist[i])
    #print(nir_filelist[i])
    #red = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_RED.TIF'
    #outFile = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_NDVI.tif'
    red=path+'/RED/'+red_filelist[i]
    
    nir=path+'/NIR/'+nir_filelist[i]
    
    outFile=path+'/NDVI/'+red_filelist[i][:-7]+'NDVI.tif'
    
    # print(red)
    # print(nir)
    # print(outFile)

    dsRed = gdal.Open(red)
    bandRed = dsRed.GetRasterBand(1)
    aRed = bandRed.ReadAsArray().astype(np.float32)

    dsNir = gdal.Open(nir)
    bandNir = dsNir.GetRasterBand(1)
    aNir = bandNir.ReadAsArray().astype(np.float32)

    ndvi = (aNir - aRed) / (aNir + aRed)

    drv = gdal.GetDriverByName('Gtiff')
    outTif = drv.Create(outFile, dsRed.RasterXSize, dsRed.RasterYSize, 1, gdal.GDT_Float32)
    outTif.SetGeoTransform(dsRed.GetGeoTransform())
    #outTif.SetProjection(ds.GetProjection())
    outTif.GetRasterBand(1).WriteArray(ndvi)
    #outTif.GetRasterBand(1).SetNoDataValue(nodataValue)
    outTif = None
# for i in  range (len(red_filelist)):
#     os.remove(path+'/RED/'+red_filelist[i])
#     os.remove(path+'/NIR/'+nir_filelist[i])

print('生產GNDVI影像')
# red=path+'/RED/'+red_filelist[i]   
# nir=path+'/NIR/'+nir_filelist[i]
# reg=path+'/REG/'+reg_filelist[i]
for i in  range (len(gre_filelist)):
    #print(red_filelist[i])
    #print(nir_filelist[i])
    #red = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_RED.TIF'
    #outFile = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_NDVI.tif'
    gre=path+'/GRE/'+gre_filelist[i]
    
    nir=path+'/NIR/'+nir_filelist[i]
    
    outFile=path+'/GNDVI/'+gre_filelist[i][:-7]+'GNDVI.tif'
    
    # print(red)
    # print(nir)
    # print(outFile)

    dsGre = gdal.Open(gre)
    bandGre = dsGre.GetRasterBand(1)
    aGre = bandGre.ReadAsArray().astype(np.float32)

    dsNir = gdal.Open(nir)
    bandNir = dsNir.GetRasterBand(1)
    aNir = bandNir.ReadAsArray().astype(np.float32)

    gndvi = (aNir - aGre) / (aNir + aGre)

    drv = gdal.GetDriverByName('Gtiff')
    outTif = drv.Create(outFile, dsGre.RasterXSize, dsGre.RasterYSize, 1, gdal.GDT_Float32)
    outTif.SetGeoTransform(dsGre.GetGeoTransform())
    #outTif.SetProjection(ds.GetProjection())
    outTif.GetRasterBand(1).WriteArray(gndvi)
    #outTif.GetRasterBand(1).SetNoDataValue(nodataValue)
    outTif = None
# for i in  range (len(red_filelist)):
#     os.remove(path+'/RED/'+red_filelist[i])
#     os.remove(path+'/NIR/'+nir_filelist[i])

print('生產GRRI影像')
for i in  range (len(red_filelist)):
    #print(red_filelist[i])
    #print(nir_filelist[i])
    #red = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_RED.TIF'
    #outFile = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_NDVI.tif'
    red=path+'/RED/'+red_filelist[i]
    
    gre=path+'/GRE/'+gre_filelist[i]
    
    outFile=path+'/GRRI/'+red_filelist[i][:-7]+'GRRI.tif'
    
    # print(red)
    # print(nir)
    # print(outFile)

    dsRed = gdal.Open(red)
    bandRed = dsRed.GetRasterBand(1)
    aRed = bandRed.ReadAsArray().astype(np.float32)

    dsGre = gdal.Open(gre)
    bandGre = dsGre.GetRasterBand(1)
    aGre = bandGre.ReadAsArray().astype(np.float32)

    grri = (aGre) / (aRed)

    drv = gdal.GetDriverByName('Gtiff')
    outTif = drv.Create(outFile, dsRed.RasterXSize, dsRed.RasterYSize, 1, gdal.GDT_Float32)
    outTif.SetGeoTransform(dsRed.GetGeoTransform())
    #outTif.SetProjection(ds.GetProjection())
    outTif.GetRasterBand(1).WriteArray(grri)
    #outTif.GetRasterBand(1).SetNoDataValue(nodataValue)
    outTif = None


# print('生產GRVI影像')
# for i in  range (len(red_filelist)):
#     #print(red_filelist[i])
#     #print(nir_filelist[i])
#     #red = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_RED.TIF'
#     #outFile = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_NDVI.tif'
#     gre=path+'/GRE/'+gre_filelist[i]
    
#     nir=path+'/NIR/'+nir_filelist[i]
    
#     outFile=path+'/GRVI/'+red_filelist[i][:-7]+'GRVI.tif'
    
#     # print(red)
#     # print(nir)
#     # print(outFile)

#     dsGre = gdal.Open(gre)
#     bandGre = dsGre.GetRasterBand(1)
#     aGre = bandGre.ReadAsArray().astype(np.float32)

#     dsNir = gdal.Open(nir)
#     bandNir = dsNir.GetRasterBand(1)
#     aNir = bandNir.ReadAsArray().astype(np.float32)

#     grvi = (aNir) / (aGre)

#     drv = gdal.GetDriverByName('Gtiff')
#     outTif = drv.Create(outFile, dsGre.RasterXSize, dsGre.RasterYSize, 1, gdal.GDT_Float32)
#     outTif.SetGeoTransform(dsGre.GetGeoTransform())
#     #outTif.SetProjection(ds.GetProjection())
#     outTif.GetRasterBand(1).WriteArray(grvi)
#     #outTif.GetRasterBand(1).SetNoDataValue(nodataValue)
#     outTif = None

# print('生產NDRE影像')
# for i in  range (len(reg_filelist)):
#     #print(red_filelist[i])
#     #print(nir_filelist[i])
#     #red = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_RED.TIF'
#     #outFile = 'F:/micasense_imageprocessing_sequoia-master/data/Sequoia/0077/IMG_180413_080658_0000_NDVI.tif'
#     reg=path+'/REG/'+reg_filelist[i]
    
#     nir=path+'/NIR/'+nir_filelist[i]
    
#     outFile=path+'/NDRE/'+reg_filelist[i][:-7]+'NDRE.tif'
    
#     # print(red)
#     # print(nir)
#     # print(outFile)

#     dsReg = gdal.Open(reg)
#     bandReg = dsReg.GetRasterBand(1)
#     aReg = bandReg.ReadAsArray().astype(np.float32)

#     dsNir = gdal.Open(nir)
#     bandNir = dsNir.GetRasterBand(1)
#     aNir = bandNir.ReadAsArray().astype(np.float32)

#     ndre = (aNir - aReg) / (aNir + aReg)

#     drv = gdal.GetDriverByName('GTiff')
#     outTif = drv.Create(outFile, dsReg.RasterXSize, dsReg.RasterYSize, 1, gdal.GDT_Float32)
#     outTif.SetGeoTransform(dsReg.GetGeoTransform())
#     #outTif.SetProjection(ds.GetProjection())
#     outTif.GetRasterBand(1).WriteArray(ndre)
#     #outTif.GetRasterBand(1).SetNoDataValue(nodataValue)
#     outTif = None

print('完成')