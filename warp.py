import cv2
import numpy as np
import os

# https://github.com/FutrCamp/OpenCV/blob/main/OpenCV/%235%20Warp%20Perspective.py


def warpping(input_path,output_path):
    filelist=os.listdir(input_path)
    for i in range (len(filelist)):

        original_image = cv2.imread(input_path+'/'+filelist[i])

        # Step 1: Define 4 corner points , use any photo editor like paint to get pixel location of the points
        #pts1 = np.float32([[85,30],[150,30],[86,125],[150,125]])
        #(左下，右下，左上，右上)[橫向，縱向]
        #初始方框座標
        if(filelist[i][-7:-4]=='NIR'):
            points_of_interest =np.float32([ [316, 655],[1044, 679],[357, 202],[1022, 216]
                     ])
        
        elif(filelist[i][-7:-4]=='RED'):
            points_of_interest =np.float32([ [361, 681],[983, 703],[397, 270],[967, 278] 
                    ])
        
        elif(filelist[i][-7:-4]=='REG'):
            points_of_interest =np.float32([ [344, 687],[1071, 712] ,[383, 233],[1050, 243]
                    ])
        
        elif(filelist[i][-7:-4]=='GRE'):
            points_of_interest =np.float32([ [398, 701],[1015, 722] ,[433, 288],[998, 295]
                    ])

        width, height = 1280, 960 # getting required size by taking care of height & width ratio of card


        # #轉換後的方框座標
        projection = np.float32([[376,675],[1084,701],[413,236],[1063,241]
                    ])

        # if(filelist[i][-7:-4]=='NIR'):
        #     points_of_interest =np.float32([ [438, 614],[961, 631],[465, 240],[946, 249]
        #              ])
        
        # elif(filelist[i][-7:-4]=='RED'):
        #     points_of_interest =np.float32([ 
        #             [429, 650],[955, 669],[460, 279],[938, 283] ])
        
        # elif(filelist[i][-7:-4]=='REG'):
        #     points_of_interest =np.float32([ [463, 641],[991, 661] ,[490, 265],[974, 273]
        #             ])

        # width, height = 1280, 960 # getting required size by taking care of height & width ratio of card


        # # #轉換後的方框座標
        # projection = np.float32([[465,649],[1041,672],[493,240],[1025,242]
        #             ])

        # Step 3: Make matrix
        matrix = cv2.getPerspectiveTransform(points_of_interest,projection)

        # Get Output image based on the above matrix
        imgOutput = cv2.warpPerspective(original_image, matrix, (width, height))
        imgOutput = cv2.rotate(imgOutput, cv2.ROTATE_180)
        cv2.imwrite(output_path+'/'+filelist[i][:-4]+'.TIF',imgOutput)
        

if __name__=='__main__':
    input_path='F:\warpping_try/a'
    output_path='F:\warpping_try/d'
    warpping(input_path,output_path)