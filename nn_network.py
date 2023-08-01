#!/usr/bin/env python
# coding: utf-8

# 

# **An approach of Neural Network to predict Iris Species**
# 
# This kernel uses multilayer perceptrons (Neural Network) to predict the species of the Iris dataset.Neural network is a machine learning algorithm which is inspired by a neuron.
# 
# ![image.png](attachment:image.png)
# 
# A neuron consists of a dendrite and an axon which are responsible for collecting and sending signals. For our artificial neural network, the concept works similar in which a lot of neurons are connected to each layer with its own corresponding weight and biases.
# 
# Although there are currently architecture of neural network, multilayer perceptron is being used as the architecture to prevent overfitting(training accuracy=good but test accuracy=bad)  to the Iris Species due to less feature.

# In[1]:

#https://www.kaggle.com/code/louisong97/neural-network-approach-to-iris-dataset/notebook

#Import required libraries 
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

# In[2]:


#Reading data 
dataset_name='19th_tomato'
train_data = pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_train/merged.csv',delimiter=",")
test_data=pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_test/merged.csv',delimiter=",")
batch_size=40
epochs=200
# data=pd.read_csv("../input/Iris.csv")
# print("Describing the data: ",data.describe())
# print("Info of the data:",data.info())

# versus_component=['red','green','blue']
# versus_component=['ndvi','gndvi','grri']
# versus_component=['red','green','blue','ndvi','gndvi','grri']
versus_component=['red','blue']
print(versus_component)
weight_path='F:\dataset/'+str(dataset_name)+'\weights/knn/'+str(versus_component) #儲存權重檔的位置
# In[3]:


# print("10 first samples of the dataset:",data.head(10))
# print("10 last samples of the dataset:",data.tail(10))


# **Visualisation of the dataset**
# 
# The coding below shows the visualisation of the dataset in order to understand the data more. It can be seen that every species of the Iris can be segregated into different regions to be predicted.




# Coding below convert the species into each respective category to be feed into the neural network

# In[5]:


# print(data["Species"].unique())


# In[6]:


# data.loc[data["Species"]=="Iris-setosa","Species"]=0
# data.loc[data["Species"]=="Iris-versicolor","Species"]=1
# data.loc[data["Species"]=="Iris-virginica","Species"]=2
# print(data.head())

train_data.loc[train_data["ripness"]=="-3","ripness"]=-3
train_data.loc[train_data["ripness"]=="-2","ripness"]=-2
train_data.loc[train_data["ripness"]=="-1","ripness"]=-1

# test_data.loc[test_data["ripness"]=="-1","ripness"]=0
# test_data.loc[test_data["ripness"]=="-2","ripness"]=1
# test_data.loc[test_data["ripness"]=="-3","ripness"]=2
# print(data.head())
#    Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm Species
# 0   1            5.1           3.5            1.4           0.2       0
# 1   2            4.9           3.0            1.4           0.2       0

# In[7]:


train_data=train_data.iloc[np.random.permutation(len(train_data))]
# print(data.head())
#       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm Species
# 20    21            5.4           3.4            1.7           0.2       0
# 88    89            5.6           3.0            4.1           1.3       1

# Converting data to numpy array in order for processing 

# In[8]:


# X=dataset.iloc[:,1:17].values
# y=dataset.iloc[:,17].values
X = train_data[versus_component].values
y = train_data['ripness'].values

# print("Shape of X",X.shape)
# print("Shape of y",y.shape)
# print("Examples of X\n",X[:3])
# print("Examples of y\n",y[:3])


# **Normalization**
# 
# It can be seen from above that the feature of the first dataset has 6cm in Sepal Length, 3.4cm in Sepal Width, 4.5cm in Petal Length and 1.6cm in Petal Width. However, the range of the dataset may be different. Therefore, in order to maintain a good accuracy, the feature of each dataset must be normalized to a range of 0-1 for processing 

# In[9]:


X_normalized=normalize(X,axis=0)
# print("Examples of X_normalised\n",X_normalized[:3])


# In[10]:


#Creating train,test and validation data
'''
80% -- train data
20% -- test data
'''
total_length=len(train_data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]

# print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
# print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])


# In[11]:


#Neural network module
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils


# In[12]:


#Change the label to one hot vector
'''
[0]--->[1 0 0]
[1]--->[0 1 0]
[2]--->[0 0 1]
'''
y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)
# print("Shape of y_train",y_train.shape)
# print("Shape of y_test",y_test.shape)
# Shape of y_train (3901, 3)
# Shape of y_test (976, 3)


# In[13]:


# model=Sequential()
# model.add(Dense(1000,input_dim=len(versus_component),activation='relu'))
# model.add(Dense(700,activation='relu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(300,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# model = Sequential()

# model.add(Dense(32, input_dim = len(versus_component), activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.4))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(3, activation = 'softmax'))
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# model.summary()

# In[14]:


# model.summary()


# In[15]:


# model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=batch_size,epochs=epochs,verbose=1)
print("Original dataset size: {}\nInput dataset size:    {}\nOutput dataset size:   {}\n".format(train_data.shape, X_normalized.shape, y.shape))
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
# seed=7
# np.random.seed(seed)
kfold = StratifiedKFold(n_splits=3, shuffle=True)
cvscores = []
for train, test in kfold.split(X_normalized, y):
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
    # model.add(Dense(12, input_dim=len(versus_component), activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

    # Fit the model
    model.fit(X_normalized[train], y[train], epochs=150, batch_size=40, verbose=0)
    # model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=batch_size,epochs=epochs,verbose=1)
    # evaluate the model
    scores = model.evaluate(X_normalized[test],y[test], verbose=0)
    print("%s: %.3f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
predictions = model.predict(X_normalized)
predictions=np.argmax(predictions,axis=1)
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
# In[16]:



prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)
# print(prediction[0:20])
# print(predict_label[0:20])
# print(y_label[0:20])

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the valuation dataset",accuracy )


# An accuracy of **100%** is achieved in this dataset.It can be asserted that for each epoch, the neural network is trying to learn from its existing feature and predict it by its weights and biases. For each epoch, the weights and biases and changed by subtracting its rate to get a better accuracy each time.
# 
# 
# **Further improvement: **
# 
# 1.Adding batch normalization 
# 
# 2.Adding dropout layer to prevent overfitting 

# In[17]:

# from keras.models import load_model
# nn_model = load_model('nn.h5')
#https://ithelp.ithome.com.tw/articles/10191627
from keras.models import model_from_json
json_string = model.to_json() 
with open(weight_path+".config", "w") as text_file:    
 text_file.write(json_string)
model.save_weights(weight_path+".weight")
from keras.models import Sequential
from keras.models import model_from_json
with open(weight_path+".config", "r") as text_file:
    json_string = text_file.read()
    nn_model = Sequential()
    nn_model = model_from_json(json_string)
    nn_model.load_weights(weight_path+".weight", by_name=False)

# with open("model.config", "w") as text_file:    
#  text_file.write(json_string)
# model.save_weights("model.weight")
# from keras.models import Sequential
# from keras.models import model_from_json
# with open("model.config", "r") as text_file:
#     json_string = text_file.read()
#     nn_model = Sequential()
#     nn_model = model_from_json(json_string)
#     nn_model.load_weights("model.weight", by_name=False)




#https://ithelp.ithome.com.tw/articles/10197575
new = test_data[versus_component].values
predict_class = test_data['ripness'].values
predict_class2 = test_data['ripness'].values
ids = test_data['id'].values

new_normalized=normalize(new,axis=0)
total_length=len(test_data)
new=new_normalized[:total_length]
predict_class=np_utils.to_categorical(predict_class ,num_classes=3)


from sklearn.decomposition import PCA
from voting import classification_voting
from sklearn import metrics
degree=len(versus_component)
# n_components=2 must be between 0 and min(n_samples, n_features)=1
# pca2 = PCA(n_components=degree, iterated_power=1)

predicted_probability=nn_model.predict(new)
length=len(new)
predicted=np.argmax(predicted_probability,axis=1)
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
# print('\nbefore voting')




# correct=0
# for i in range(len(predict_class)):
#     if(predicted[i]==predict_class[i]):
#         correct=correct+1
# print('Test  Accuracy:%.2f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))
# fuzzy_point=0.5
# correct=0
# for i in range(len(predict_class)):
#     if(predicted[i]==predict_class[i]):
#         correct=correct+1
#     elif(abs(predicted[i]-predict_class[i])==1):
#         correct=correct+fuzzy_point

# print('Test  Accuracy(wide +-1):%.2f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))


# for i in range(len(predict_class)):
#     if(int(predict_class[i])<=-2):
#         predict_class2[i]=-2
#     if(int(predicted[i])<=-2):
#         predicted[i]=-2

# correct=0
# for i in range(len(predict_class2)):
#     if(predicted[i]==predict_class2[i]):
#         correct=correct+1
    

# print('Test  Accuracy(only ripe or not)(no+-1):%.2f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))

# In[4]:


# sns.lmplot('SepalLengthCm', 'SepalWidthCm',
#            data=data,
#            fit_reg=False,
#            hue="Species",
#            scatter_kws={"marker": "D",
#                         "s": 50})
# plt.title('SepalLength vs SepalWidth')

# sns.lmplot('PetalLengthCm', 'PetalWidthCm',
#            data=data,
#            fit_reg=False,
#            hue="Species",
#            scatter_kws={"marker": "D",
#                         "s": 50})
# plt.title('PetalLength vs PetalWidth')

# sns.lmplot('SepalLengthCm', 'PetalLengthCm',
#            data=data,
#            fit_reg=False,
#            hue="Species",
#            scatter_kws={"marker": "D",
#                         "s": 50})
# plt.title('SepalLength vs PetalLength')

# sns.lmplot('SepalWidthCm', 'PetalWidthCm',
#            data=data,
#            fit_reg=False,
#            hue="Species",
#            scatter_kws={"marker": "D",
#                         "s": 50})
# plt.title('SepalWidth vs PetalWidth')
# plt.show()