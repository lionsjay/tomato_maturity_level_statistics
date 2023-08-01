import os
#變更檔案名稱順序用
path='F:/dataset/lin_tomato/sequoia_mot/3rd'
#path='F:/dataset/19th_tomato/sequoia_mot/6th'


img_filelist=os.listdir(path+'/img1')
for i in range (len(img_filelist)):
    os.rename(path+'/img1/'+img_filelist[i],path+'/img1/'+str(i).rjust(6,'0')+'.jpg')
    
labels_filelist=os.listdir(path+'/labels_with_ids')
for i in range (len(labels_filelist)):
    if not os.path.isdir(path+'/gt/'):
        os.mkdir(path+'/gt/')
    os.rename(path+'/labels_with_ids/'+labels_filelist[i],path+'/labels_with_ids/'+str(i).rjust(6,'0')+'.txt')
