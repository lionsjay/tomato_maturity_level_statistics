import fitz
import os
import glob
from reportlab.pdfgen import canvas
from PIL import Image

def png2pdf(root):
    for img in sorted(glob.glob("{}/*.[pP][nN][gG]".format(root))):
        img_name = img.split('\\')[-1].split('.')[0]
        doc = fitz.open()
        imgdoc = fitz.open(img)
        pdfbytes = imgdoc.convertToPDF()
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insertPDF(imgpdf)
        doc.save(root+'\\'+img_name+'.pdf')
        doc.close()
        print('Convert '+img_name+'.png'+' to ' +img_name+'.pdf')

    for img in sorted(glob.glob("{}/*.[jJ][pP][gG]".format(root))):
        img_name = img.split('\\')[-1].split('.')[0]
        doc = fitz.open()
        imgdoc = fitz.open(img)
        pdfbytes = imgdoc.convertToPDF()
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insertPDF(imgpdf)
        doc.save(root+'\\'+img_name+'.pdf')
        doc.close()
        print('Convert '+img_name+'.JPG'+' to ' +img_name+'.pdf')

    # for img in sorted(glob.glob("{}/*.TIF".format(root))):
    #     img_name = img.split('\\')[-1].split('.')[0]
    #     print(img)
    #     imgtif = Image.open(img)
    #     # Create a PDF document
    #     pdf_file = canvas.Canvas(img_name+".pdf")
    #     # Draw the image on the PDF
    #     pdf_file.drawImage(imgtif, 0, 0)
    #     # Save the PDF
    #     pdf_file.save()
    #     print('Convert '+img_name+'.TIF'+' to ' +img_name+'.pdf')


if __name__=='__main__':
   # path='E:/110白筑安/try'
   path='E:/110白筑安/images'
   
   png2pdf(path)
