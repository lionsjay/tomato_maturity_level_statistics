import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#https://blog.csdn.net/weixin_43474731/article/details/100155593
#https://www.delftstack.com/zh-tw/howto/python/python-write-to-an-excel-spreadsheet/

def excel_clear(file_name):
    #准备一个有空格的excle文件
    # file_name = r'F:\warpping_try/blank.xls'
    # file_name = r'first.xlsx'
    #读取EXCLE文件
    data = pd.read_excel(file_name)
    
    
    # datanota = data[data['e'].notna()]
    # datanota2 = datanota[data['w'].notna()]
    datanota = data[data['ndvi'].notna()]
    datanota2 = datanota[data['grri'].notna()]
    datanota3 = datanota2[data['gndvi'].notna()]
    datanota4 = datanota3[data['ripness'].notna()]
    datanota5 = datanota4[data['green'].notna()]
    # print(datanota2)
    datanota5.to_excel(file_name, sheet_name='sheet1', index=False)
    # datanota2.to_excel('F:\warpping_try/sample.xls', sheet_name='sheet1', index=False)

def predict_excel_clear(file_name):

    data = pd.read_excel(file_name)
    
    
    # datanota = data[data['e'].notna()]
    # datanota2 = datanota[data['w'].notna()]
    datanota = data[data['ndvi'].notna()]
    datanota2 = datanota[data['grri'].notna()]
    datanota3 = datanota2[data['gndvi'].notna()]
    datanota4 = datanota3[data['green'].notna()]
    # print(datanota2)
    datanota4.to_excel(file_name, sheet_name='sheet1', index=False)
    # datanota2.to_excel('F:\warpping_try/sample.xls', sheet_name='sheet1', index=False)

if __name__=='__main__':
    file_name=r'F:\warpping_try/blank.xls'
    excel_clear(file_name)