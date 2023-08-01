#https://towardsdatascience.com/image-processing-with-python-applying-homography-for-image-warping-84cd87d2108f
#Image Processing with Python — Applying Homography for Image Warping

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import transform
from skimage.io import imread, imshow

# input_path='F:\warpping_try/a'
# output_path='F:\warpping_try/d'


def warpping(input_path,output_path):
    filelist=os.listdir(input_path)
    for i in range (len(filelist)):

        original_image = imread(input_path+'/'+filelist[i])
        # original_image = imread('F:\warpping_try/a/IMG_221103_075519_0000_RED.tif')
        # plt.figure(num=None, figsize=(8, 6), dpi=80)

        #(左上，右上，右下，左下)[橫向，縱向]
        #初始方框座標

        if(filelist[i][-7:-4]=='NIR'):
            points_of_interest =[[383, 468], 
                                [969, 347], 
                                [1006, 643], 
                                [411, 768]]
        
        elif(filelist[i][-7:-4]=='RED'):
            points_of_interest =[[397, 531], 
                                [940, 414], 
                                [976, 699], 
                                [424, 818]]
        
        elif(filelist[i][-7:-4]=='REG'):
            points_of_interest =[[420, 502], 
                                [994, 382], 
                                [1030, 673], 
                                [447, 800]]

        # #轉換後的方框座標
        projection = [[420, 490], 
                                [1015, 361], 
                                [1057, 673], 
                                [446, 802]]

        


        # #轉換後的方框座標
        # projection = [
        #                 [277, 201], 
        #                 [673, 204],
        #                 [677, 504], 
        #                 [291, 510]
                        
                        
        #                 ]


        # if(filelist[i][-7:-4]=='NIR'):
        #     points_of_interest =[[622.08, 231.36], 
        #                         [701.44, 231.36], 
        #                         [701.44, 396.48], 
        #                         [622.08, 396.48]]
        
        # elif(filelist[i][-7:-4]=='RED'):
        #     points_of_interest =[[622.08, 231.36], 
        #                         [701.44, 231.36], 
        #                         [701.44, 396.48], 
        #                         [622.08, 396.48]]
        
        # elif(filelist[i][-7:-4]=='REG'):
        #     points_of_interest =[[622.08, 231.36], 
        #                         [701.44, 231.36], 
        #                         [701.44, 396.48], 
        #                         [622.08, 396.48]]

        


        # # #轉換後的方框座標
        # projection = [[569.6, 254.4], 
        #                 [648.96, 254.4], 
        #                 [648.96, 419.52], 
        #                 [569.6, 419.52]]

        # #初始方框座標
        # points_of_interest =[[105, 60], 
        #                        [260, 85], 
        #                        [275, 295], 
        #                        [110, 290]]


        # # #轉換後的方框座標
        # projection = [[305, 260], 
        #                 [460, 285], 
        #                 [475, 495], 
        #                 [310, 490]]
        color = 'green'
        patches = []

        # fig, ax = plt.subplots(1,2, figsize=(15, 10), dpi = 80)
        # for coordinates in (points_of_interest + projection):
        #     patch = Circle((coordinates[0],coordinates[1]), 10, 
        #                     facecolor = color)
        #     patches.append(patch)
        # for p in patches[:4]:
        #     ax[0].add_patch(p)
        # ax[0].imshow(original_image)
        # for p in patches[4:]:
        #     ax[1].add_patch(p)
        # ax[1].imshow(np.ones((original_image.shape[0], original_image.shape[1])))

        points_of_interest = np.array(points_of_interest)
        projection = np.array(projection)
        tform = transform.estimate_transform('projective', points_of_interest, projection)
        tf_img_warp = transform.warp(original_image, tform.inverse, mode = 'edge')
        # plt.figure(num=None, figsize=(8, 6), dpi=80)
        # fig, ax = plt.subplots(1,2, figsize=(15, 10), dpi = 80)
        # ax[0].set_title(f'Original', fontsize = 15)
        # ax[0].imshow(original_image)
        # ax[0].set_axis_off()
        # ax[1].set_title(f'Transformed', fontsize = 15)
        # ax[1].imshow(tf_img_warp)
        # ax[1].set_axis_off()
        plt.imsave(output_path+'/'+filelist[i][:-4]+'.tiff',tf_img_warp)
        
        # tf_img_warp=np.sin(tf_img_warp)
        # plt.figure()
        # plt.plot(tf_img_warp)
        # plt.savefig(output_path+'/'+filelist[i][:-4]+'.TIF')
        
        # plt.imsave('F:\warpping_try/Transformed2.tiff',tf_img_warp)

    # for i in range (len(filelist)):
    #     if(filelist[i][-4:]=='.TIF' or filelist[i][-4:]=='.tif'):
    #         os.remove(input_path+'/'+filelist[i])


    #plt.show()

if __name__=='__main__':
    input_path='F:\warpping_try/a'
    output_path='F:\warpping_try/d'
    warpping(input_path,output_path)