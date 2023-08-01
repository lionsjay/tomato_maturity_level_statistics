import matplotlib.pyplot as plt
import scipy.io
import cv2
import os
from tqdm import tqdm
import micasense.metadata as metadata
import micasense.sequoiautils as msutils
    
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('matplotlib inline')

def calibration(imageName):
    
    # Read raw image DN values
    # reads 16 bit tif - this will likely not work for 12 bit images
    imageRaw=plt.imread(imageName)
       
    # Optional: pick a color map that fits your viewing style
    # one of 'gray, viridis, plasma, inferno, magma, nipy_spectral'

    exiftoolPath = None
    if os.name == 'nt':
        exiftoolPath = 'C:/Users/USER/OneDrive/桌面/exiftool.exe'
    # get image metadata
    meta = metadata.Metadata(imageName, exiftoolPath=exiftoolPath)
    bandName = meta.get_item('XMP:BandName')

    SequoiaIrradiance, V = msutils.sequoia_irradiance(meta, imageRaw)
    
    # Sunshine sensor Irradiance
    SunIrradiance = msutils.GetSunIrradiance(meta)
    
    # Light calibrated sequoia irradiance
    SequoiaIrradianceCalibrated = SequoiaIrradiance/SunIrradiance
    
    markedImg = SequoiaIrradianceCalibrated.copy()
    if bandName == 'NIR':
        ulx = 622 # upper left column (x coordinate) of panel area
        uly = 605 # upper left row (y coordinate) of panel area
        lrx = 707 # lower right column (x coordinate) of panel area
        lry = 675 # lower right row (y coordinate) of panel area
    elif bandName == 'Red':
        ulx = 620 # upper left column (x coordinate) of panel area
        uly = 570 # upper left row (y coordinate) of panel area
        lrx = 703 # lower right column (x coordinate) of panel area
        lry = 638 # lower right row (y coordinate) of panel area
    elif bandName == 'Green':
        ulx = 590 # upper left column (x coordinate) of panel area
        uly = 560 # upper left row (y coordinate) of panel area
        lrx = 675 # lower right column (x coordinate) of panel area
        lry = 635 # lower right row (y coordinate) of panel area

    cv2.rectangle(markedImg,(ulx,uly),(lrx,lry),(0,255,0),3)
    
    # Our panel calibration by band (from MicaSense for our specific panel)
    panelCalibration = {  
        "Green": 0.186, 
        "Red": 0.199, 
        "Red edge": 0.229, 
        "NIR": 0.263 
    }
    
    # Select panel region from radiance image
    panelRegion = SequoiaIrradianceCalibrated[uly:lry, ulx:lrx]
    meanRadiance = panelRegion.mean()
    panelReflectance = panelCalibration[bandName]
    radianceToReflectance = panelReflectance / meanRadiance
    
    return radianceToReflectance, meta

def read(flightImageName, radianceToReflectance, meta, target, matName):

    flightImageRaw=plt.imread(flightImageName)
    
    flightRadianceImage, _ = msutils.sequoia_irradiance(meta, flightImageRaw)
    
    # Sunshine sensor Irradiance
    flightSunIrradiance = msutils.GetSunIrradiance(meta)
    
    # Light calibrated sequoia irradiance
    flightSequoiaIrradianceCalibrated = flightRadianceImage/flightSunIrradiance
    
    flightReflectanceImage = flightSequoiaIrradianceCalibrated * radianceToReflectance
    flightUndistortedReflectance = msutils.correct_lens_distortion_sequoia(meta, flightReflectanceImage)

    scipy.io.savemat(os.path.join(target, matName), mdict={matName[-7:-4]:flightUndistortedReflectance})


def main(path,mode):
    # calibration
    print('影像校正中...')
    imagePath = os.path.join(path,'calibration')
    # imagePath = os.path.join('.','data','Sequoia','0081')
    for file in os.listdir(imagePath):
        imageName = os.path.join(imagePath,file)
        if imageName[-7:-4] == 'NIR':
            radianceToReflectance_NIR, meta_NIR = calibration(imageName)
        elif imageName[-7:-4] == 'RED':
            radianceToReflectance_RED, meta_RED = calibration(imageName)
        elif imageName[-7:-4] == 'GRE':
            radianceToReflectance_GRE, meta_GRE = calibration(imageName)
    print('影像校正完成')
    # processing        
    #magePath = os.path.join('.','data','Sequoia','image','0014_spectral')
    # imagePath = os.path.join('.','data','Sequoia','0081')  
    # mode='3l_train'      
    imagePath = os.path.join(path,mode)  
    pbar = tqdm(total=len(imagePath),desc='轉換成mat檔')
    for file in os.listdir(imagePath):
        imageName = os.path.join(imagePath,file)
        matName = file[:-4] + '.mat'
        #target = os.path.join('.','data','Sequoia','result')
        # target = os.path.join('.','data','Sequoia','0082')
        
        nir_target = os.path.join(path,mode,'NIR')
        red_target = os.path.join(path,mode,'RED')
        gre_target = os.path.join(path,mode,'GRE')
        if not os.path.exists(nir_target):
            os.makedirs(nir_target)
        if not os.path.exists(red_target):
            os.makedirs(red_target)
        if not os.path.exists(red_target):
            os.makedirs(nir_target)
        if imageName[-7:-4] == 'NIR':
            read(imageName, radianceToReflectance_NIR, meta_NIR, nir_target, matName)
            pbar.update()
        elif imageName[-7:-4] == 'RED':
            read(imageName, radianceToReflectance_RED, meta_RED, red_target, matName)
            pbar.update()
        elif imageName[-7:-4] == 'GRE':
            read(imageName, radianceToReflectance_GRE, meta_GRE, gre_target, matName)
            pbar.update()
        # print('\r{} done'.format(file[:-4]), end = '')
    pbar.close()

if __name__ =="__main__":
 path = os.path.join('.','19th_tomato','4th','Sequoia')
 mode='5l_train' 
 main(path,mode)

