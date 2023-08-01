import os

dataset_path='F:/dataset/'
img_path='19th_tomato/7th/ipad/5l_train/images'#左邊不要有底線
#img_path2='/home/lionsjay/Desktop/dataset/itzu_tomato_datasets/video1/images'
txt_path='F:/code/FairMOT-master/src/data/19th_tomato'
filelist = os.listdir(dataset_path+img_path)#讀取該路徑下的所有檔案名稱(字串)



#要val
train=open(txt_path+'.train','a+')
# if os.path.isfile(train):
#     train.truncate(0)
val=open(txt_path+'.val','a+')
# if os.path.isfile(val):
#     val.truncate(0)

for i in range (len(filelist)):
    if ((i%20)>=15):
        val.write(img_path+'/'+filelist[i]+'\n')  #/home/lionsjay/Desktop/dataset/itzu_tomato_datasets/IMG8213/images/000001.jpg
    else:
        train.write(img_path+'/'+filelist[i]+'\n')


# #不要val
# train=open(txt_path+'.train','a+')
# # if os.path.isfile(train):
# #     train.truncate(0)
# for i in range (len(filelist)):
#      train.write(img_path+'/'+filelist[i]+'.jpg'+'\n')
