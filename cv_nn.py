
# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# fix random seed for reproducibility
seed = 7
# numpy.random.seed(seed)
# load pima indians dataset
dataset_name='19th_tomato'
dataset = numpy.loadtxt('F:\dataset/'+str(dataset_name)+'\svm_train/merged.csv',delimiter=",",skiprows=1)
dataset_name='19th_tomato'
# train_data = pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_train/merged.csv',delimiter=",")
versus_component=['red','green','blue']
# versus_component=['ndvi','gndvi','grri']
# versus_component=['red','green','blue','ndvi','gndvi','grri']
# versus_component=['red','blue']
# split into input (X) and output (Y) variables
# total_length=len(dataset)
# X = train_data[versus_component].values
# Y = train_data['ripness'].values
X = dataset[:,1:4]
Y = dataset[:,17]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
for train, test in kfold.split(X, Y):
  # create model
    model = Sequential()
    model.add(Dense(32, input_dim = len(versus_component), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=40, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
import numpy as np
prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)
# print(prediction[0:20])
# print(predict_label[0:20])
# print(y_label[0:20])

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the valuation dataset",accuracy )