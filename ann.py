#https://github.com/architsingh15/Keras-Neural-Network-Analysis-Iris-Dataset/blob/master/code.py

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Activation,Dropout 
from tqdm import tqdm
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

seed = 42
numpy.random.seed(seed)
dataset_name='19th2_tomato'
train_data = pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_train/merged.csv',delimiter=",")
test_data=pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_test/merged.csv',delimiter=",")
# train_data = train_data.values
fold=5

# versus_component=['red','green','blue']
# versus_component=['ndvi','gndvi','grri']
# versus_component=['red','green','blue','ndvi','gndvi','grri']
versus_component=['red','blue']
weight_path='F:\dataset/'+str(dataset_name)+'\weights/ann/'+str(versus_component) #儲存權重檔的位置
print(dataset_name)
print(versus_component)
X = train_data[versus_component].values
Y = train_data['ripness'].values

# X = train_data[:,0:4].astype(float)
# Y = train_data[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
start=time.time()
pbar = tqdm(total=fold,desc='人工神經網路訓練中')
def baseline_model():

	
	# model.add(Dense(4, input_dim=len(versus_component), kernel_initializer='normal', activation='relu'))
	# model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))

	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = Sequential()
    model.add(Dense(300,input_dim=len(versus_component),activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(700,activation='relu'))
    # model.add(Dense(1000,input_dim=len(versus_component),activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(1500,activation='relu'))
    model.add(Dense(1000,activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(700,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300,activation='relu'))    
    model.add(Dense(3,activation='softmax'))

    # model.add(Dense(32, input_dim = len(versus_component), activation = 'relu'))
    # model.add(Dense(64, activation = 'relu'))
    # model.add(Dense(128, activation = 'relu'))
    # model.add(Dense(64, activation = 'relu'))
    # model.add(Dense(32, activation = 'relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(16, activation = 'relu'))
    # model.add(Dense(3, activation = 'softmax'))    
    model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
    # model.summary()
    pbar.update()
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=300, batch_size=40, verbose=0)
kfold = KFold(n_splits=fold, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
pbar.close()
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
estimator.fit(X, dummy_y)
# https://stackoverflow.com/questions/44622857/why-am-i-getting-attributeerror-kerasclassifier-object-has-no-attribute-mode
# https://blog.csdn.net/Mr_green_bean/article/details/88600497
end=time.time()

print('訓練時間:'+str(end-start)+'秒')

#  model_tt.model.save("test_model.h5")

# print (model_tt.score(X_test, y_test))
# print (model_tt.predict_proba(X_test))
# print (model_tt.predict(X_test))

#https://ithelp.ithome.com.tw/articles/10191627
# from keras.models import model_from_json
# json_string = estimator.to_json() 
# with open(weight_path+".config", "w") as text_file:    
#  text_file.write(json_string)
# estimator.save_weights(weight_path+".weight")
# from keras.models import Sequential
# from keras.models import model_from_json
# with open(weight_path+".config", "r") as text_file:
#     json_string = text_file.read()
#     nn_model = Sequential()
#     nn_model = model_from_json(json_string)
#     nn_model.load_weights(weight_path+".weight", by_name=False)

from keras.models import load_model
estimator.model.save(weight_path+'.h5')  # creates a HDF5 file 'model.h5'
from keras.models import load_model

# # 刪除既有模型變數
# del estimator 

# 載入模型
estimator.model = load_model(weight_path+'.h5')


import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#https://ithelp.ithome.com.tw/articles/10197575
new = test_data[versus_component].values
predict_class = test_data['ripness'].values
predict_class2 = test_data['ripness'].values
ids = test_data['id'].values

total_length=len(test_data)

predict_class=np_utils.to_categorical(predict_class ,num_classes=3)


from sklearn.decomposition import PCA
from voting import classification_voting
from sklearn import metrics
degree=len(versus_component)
# n_components=2 must be between 0 and min(n_samples, n_features)=1
# pca2 = PCA(n_components=degree, iterated_power=1)

# predicted_probability=estimator.predict(new)
predicted=estimator.predict(new)
length=len(new)
# predicted=np.argmax(predicted_probability,axis=1)
# print('voting')
# predicted=classification_voting(predicted,ids)


# c_matrix = metrics.confusion_matrix(predict_class,predicted)
# print(c_matrix)

# print(predicted[0:30])
# predicted_label2=np.argmax(predict_class,axis=1)
# print(predicted_label2[0:30])

predict_class=np.argmax(predict_class,axis=1)
# print(predict_class[0:30])
accuracy=np.sum(predicted==predict_class)/length * 100 
print("Accuracy of the test dataset",accuracy )

# seed = 10
# numpy.random.seed(seed)

# dataframe = pd.read_csv("Iris.csv", header=None)
# dataset = dataframe.values
# X = dataset[1:,0:4].astype(float)
# Y = dataset[1:,4]

# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# dummy_y = np_utils.to_categorical(encoded_Y)

# def baseline_model():

# 	model = Sequential()
# 	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))

# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model

# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))