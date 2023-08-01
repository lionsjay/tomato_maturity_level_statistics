from pdf2image import convert_from_path
import glob
path='E:/110白筑安\研討會\ieee_images'

for img in sorted(glob.glob("{}/*.[pP][dD][fF]".format(path))):
        img_name = img.split('\\')[-1].split('.')[0]
        pages = convert_from_path(img, 50)
        for page in pages:
            page.save(path+'\\'+img_name+'.png', 'PNG')
        print('Convert '+img_name+'.pdf'+' to ' +img_name+'.png')
        # pages.save(path+'\\'+img_name+'.png', 'PNG')
# for page in pages:
#     page.save(page+'.png', 'PNG')