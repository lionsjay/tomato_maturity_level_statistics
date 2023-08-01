import os
import pandas as pd
import numpy as np
import openpyxl
#https://www.796t.com/article.php?id=106723


dir = "F:/dataset/19th_tomato\svm_test"#設定工作路徑
#新建列表，存放檔名（可以忽略，但是為了做的過程能心裡有數，先放上）
filename_excel = []
#新建列表，存放每個檔案資料框（每一個excel讀取後存放在資料框）
frames = []
min_id = 0
# workbook = openpyxl.load_workbook(excel_path)
# sheet = workbook.worksheets[0]

for root,dirs,files in os.walk(dir):
  for file in files:
    #print(os.path.join(root,file))
    filename_excel.append(os.path.join(root,file))
    df = pd.read_excel(os.path.join(root,file)) #excel轉換成DataFrame
    y = df['id'].values
    
    # print(df['id'].values[y])
    for i in range(len(y)):
      y[i]=y[i]+min_id
      
      
      
    pd.DataFrame(y).to_excel(file,header=True,index=False)
    

    frames.append(df)
    # print(min_id + np.max(y),min_id,np.max(y))
    min_id = np.max(y) 
    
    
#列印檔名
print(filename_excel) 
 #合併所有資料
result = pd.concat(frames) 
result.sort_values(by='id',inplace=True,ascending=True) #ascending 默认等于True，按从小到大排列，改为False 按从大到小排
#檢視合併後的資料
result.head()
result.shape
result.to_csv(dir+'/'+'merged.csv',sep=',',index = False)#儲存合併的資料到電腦D盤的merge資料夾中，並把合併後的檔案命名為a12.csv
# result.to_excel(excel_path+'/'+'merged.xls',sep=',',index = False)#儲存合併的資料到電腦D盤的merge資料夾中，並把合併後的檔案命名為a12.csv