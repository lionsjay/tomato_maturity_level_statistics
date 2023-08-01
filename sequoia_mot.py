import os
import tqdm
path='F:\dataset/19th2_tomato\sequoia_mot/9th'
#path='F:/dataset/lin_tomato/sequoia_mot/2nd'
cuhksysu_path=path+'/labels_with_ids/'
#1,7,279,782,113,116,1,-1,-1,-1
mot_path=path+'/gt/'
#0 2 0.371296 0.448698 0.0574074 0.0401042
if not os.path.isdir(mot_path):
        os.mkdir(mot_path)

print(path)
width=4608
height=3456
# width=1920
# height=1080
filelist=os.listdir(cuhksysu_path)
#將檔案名稱改成數字
img_filelist=os.listdir(path+'/img1')
labels_filelist=os.listdir(path+'/labels_with_ids')
if not os.path.isdir(path+'/gt/'):
        os.mkdir(path+'/gt/')
for i in range (len(img_filelist)):
    os.rename(path+'/img1/'+img_filelist[i],path+'/img1/'+str(i).rjust(6,'0')+'.jpg')
    try:
        os.rename(path+'/labels_with_ids/'+img_filelist[i][:-4]+'.txt',path+'/labels_with_ids/'+str(i).rjust(6,'0')+'.txt')
    except:
        f = open(path+'/labels_with_ids/'+str(i).rjust(6,'0')+'.txt', 'w')
        f.close()
        continue
    
    
# labels_filelist=os.listdir(path+'/labels_with_ids')
# for i in range (len(labels_filelist)):
#     if not os.path.isdir(path+'/gt/'):
#         os.mkdir(path+'/gt/')
#     os.rename(path+'/labels_with_ids/'+labels_filelist[i],path+'/labels_with_ids/'+str(i).rjust(6,'0')+'.txt')

# #製作mot檔
filelist=os.listdir(cuhksysu_path)
with open(mot_path+'gt.txt','w') as g:#
#g = open(mot_path+'gt.txt','a+')
        for i in range (len(filelist)):
                f = open(cuhksysu_path+'/'+filelist[i])
                # print(cuhksysu_path+'/'+filelist[i])
                # lines = len(f.readlines())
                # print(lines)
                for line in f.readlines():
                        # print(line)
                        s = line.split(' ')
                        #0 2 0.371296 0.448698 0.0574074 0.0401042
                        #cuhk-sysu_data_fmt: [c=0, id, ncx, ncy, nw, nh]
                        # print(filelist[i])
                        frame=int(filelist[i][:-4])
                        ids=s[1]
                        ncx=float(s[2])
                        ncy=float(s[3])
                        nw=float(s[4])
                        nh=float(s[5])
                        x1=(ncx-0.5*nw)*width
                        y1=(ncy-0.5*nh)*height
                        w=nw*width
                        h=nh*height
                        g.write((str(frame)+','+ids+','+str(int(x1))+','+str(int(y1))+','+str(int(w))+','+str(int(h))+',1,-1,-1,-1\n'))        
                        #mot_data_fmt: [fn, id, x1, y1, w, h, c=1, c=-1, c=-1, c=-1]
        
#g.close()

#製作seqinfo.ini
number=path.split('/')
with open(path+'/seqinfo.ini','w') as h:
        h.write('[Sequence]\n')
        h.write('name='+number[-1]+'\n')
        h.write('frameRate=1\n')
        h.write('seqLength='+str(len(img_filelist))+'\n')
        h.write('imWidth='+str(width)+'\n')
        h.write('imHeight='+str(height)+'\n')
        h.write('imExt=.jpg')
        # [Sequence]
        # name=2nd
        # imDir=img1
        # frameRate=1
        # seqLength=103
        # imWidth=4608
        # imHeight=3456
        # imExt=.jpg
