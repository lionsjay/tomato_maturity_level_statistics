#https://www.kaggle.com/code/skalskip/iris-data-visualization-and-knn-classification
import numpy as np
import pandas as pd
import sys
from voting import classification_voting
from sklearn.preprocessing import normalize #machine learning algorithm library
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

dataset_name='lin_tomato'
dataset = pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_train/merged.csv',delimiter=",")
test_data=pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_test/merged.csv',delimiter=",")

pca_path='F:\dataset/'+str(dataset_name)+'\pca/' #F:\dataset/19th_tomato\pca\

#dataset = pd.read_csv('../input/Iris.csv')
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
dataset.shape
dataset.head(5)
dataset.describe()
# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
dataset.groupby('ripness').size()
# dataset.groupby('Species').size()

# versus_component=['red','green','blue']
# versus_component=['ndvi','gndvi','grri']
versus_component=['red','green','blue','ndvi','gndvi','grri']
# versus_component=['red','blue']
# versus_component=['hue','saturation','brightness']
# versus_component=['red','green','r-g','g-b','hue','saturation','brightness','s-h','l','a','b','l-a']
#feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']

print(versus_component)
weight_path='F:\dataset/'+str(dataset_name)+'\weights/knn/'+str(versus_component) #儲存權重檔的位置


X = dataset[versus_component].values
y = dataset['ripness'].values
# x_normalized=normalize(X,axis=0)
# y = dataset['Species'].values

# Alternative way of selecting features and labels arrays:
# X = dataset.iloc[:, 1:5].values
# y = dataset.iloc[:, 5].values
from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
from sklearn.model_selection import train_test_split #https://blog.csdn.net/qq_35962520/article/details/85295228
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
reverse_versus_component=['id','red','green','blue','r-g','hue','saturation','brightness','s-h','l','a','b','l-a','ndvi','ndre','gndvi','grri']
i=0
while i < len(reverse_versus_component):
    for j in range(len(versus_component)):
        #print(i,j)
        if(versus_component[j]==reverse_versus_component[i]):
            reverse_versus_component.pop(i)
            i=i-1
    i=i+1

# print('reverse_versus_component='+str(reverse_versus_component))
# 可視化
# print('繪製可視化圖')
# from pandas.plotting import parallel_coordinates
# plt.figure(figsize=(15,10))
# parallel_coordinates(dataset.drop("id", axis=1), "ripness")
# plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Features values', fontsize=15)
# plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
# plt.savefig(pca_path+'Parallel Coordinates Plot.png')
# # plt.show()
# # from pandas.plotting import andrews_curves
# # plt.figure(figsize=(15,10))
# # andrews_curves(dataset.drop("id", axis=1), "ripness")
# # plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
# # plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
# # plt.savefig(pca_path+'Andrews Curves Plot.png')
# # plt.show()
# # plt.figure()
# # sns.pairplot(dataset.drop("id", axis=1), hue = "ripness", size=3, markers=["o", "s", "D","+"])
# sns.pairplot(dataset.drop(reverse_versus_component, axis=1), hue = "ripness",palette="husl", size=len(versus_component),markers=[ 'o', 'v', '*'])  #可使用[',', '.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
# plt.savefig(pca_path+'/'+str(versus_component)+'.png')
# # plt.show()
# #dataset.drop語法 https://blog.csdn.net/songyunli1111/article/details/79306639
# plt.figure()
# dataset.drop(reverse_versus_component, axis=1).boxplot(by="ripness", figsize=(15, 10))
# plt.savefig(pca_path+str(versus_component)+' Boxplot grouped by ripness.png')
# print('可視化圖繪製完畢')
# plt.show()



# Loading libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Building confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Calculating model accuracy
accuracy = accuracy_score(y_test, y_pred)*100
# print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
# creating list of K for KNN
k_list = list(range(2,30,2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# plt.figure()
# plt.figure(figsize=(15,10))
# plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
# plt.xlabel('Number of Neighbors K', fontsize=15)
# plt.ylabel('Misclassification Error', fontsize=15)
# sns.set_style("whitegrid")
# plt.plot(k_list, MSE)

# plt.show()

# finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

import numpy as np
import pandas as pd
import scipy as sp

class MyKNeighborsClassifier():
    """
    My implementation of KNN algorithm.
    """
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        
    def fit(self, X, y):
        """
        Fit the model using X as array of features and y as array of labels.
        """
        n_samples = X.shape[0]
        # number of neighbors can't be larger then number of samples
        if self.n_neighbors > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        # finding and saving all possible class labels
        self.classes_ = np.unique(y)
        
        self.X = X
        self.y = y
        
    def predict(self, X_test):
        
        # number of predictions to make and number of features inside single sample
        n_predictions, n_features = X_test.shape
        
        # allocationg space for array of predictions
        predictions = np.empty(n_predictions, dtype=int)
        
        # loop over all observations
        for i in range(n_predictions):
            # calculation of single prediction
            predictions[i] = single_prediction(self.X, self.y, X_test[i, :], self.n_neighbors)

        return(predictions)
def single_prediction(X, y, x_train, k):
    
    # number of samples inside training set
    n_samples = X.shape[0]
    
    # create array for distances and targets
    distances = np.empty(n_samples, dtype=np.float64)

    # distance calculation
    for i in range(n_samples):
        distances[i] = (x_train - X[i]).dot(x_train - X[i])
    
    # combining arrays as columns
    distances = sp.c_[distances, y]
    # sorting array by value of first column
    sorted_distances = distances[distances[:,0].argsort()]
    # celecting labels associeted with k smallest distances
    targets = sorted_distances[0:k,1]

    unique, counts = np.unique(targets, return_counts=True)
    return(unique[np.argmax(counts)])
# Instantiate learning model (k = 3)
# my_classifier = MyKNeighborsClassifier(n_neighbors=3)
# my_classifier = MyKNeighborsClassifier(n_neighbors=best_k)
my_classifier = KNeighborsClassifier(n_neighbors=best_k)
# my_classifier = KNeighborsClassifier(n_neighbors=best_k)

# Fitting the model
my_classifier.fit(X_train, y_train)

# Predicting the Test set results
my_y_pred = my_classifier.predict(X_test)
accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
from sklearn import metrics
c_matrix = metrics.confusion_matrix(y_test,my_y_pred)
 
print(c_matrix)





import joblib #https://ithelp.ithome.com.tw/articles/10197575
joblib.dump(my_classifier,weight_path+'.pkl')#儲存權重檔
# joblib.dump(classifier,'knn.pkl')#儲存權重檔
clf2 = joblib.load(weight_path+'.pkl')

#https://ithelp.ithome.com.tw/articles/10197575
new = test_data[versus_component].values
# new=normalize(new,axis=0)
predict_class = test_data['ripness'].values
predict_class2 = test_data['ripness'].values
ids = test_data['id'].values
# print(ids[0:300])
# print(predict_class)
# print(predict_class)
#[[231.9162695  118.2381781   87.32090492   7.29249437]
# [185.3584926  182.6752758  130.7640943   29.04894868]]
from sklearn.decomposition import PCA

degree=len(versus_component)
# n_components=2 must be between 0 and min(n_samples, n_features)=1
pca2 = PCA(n_components=degree, iterated_power=1)
# pca = PCA(n_components=2, iterated_power=1)
test_reduced = pca2.fit_transform(new)
predicted=clf2.predict(new)
predicted=classification_voting(predicted,ids)
print('voting')

c_matrix = metrics.confusion_matrix(predict_class,predicted)
print(c_matrix)
c_matrix.tolist()
# comfusion_matrix=np.array(c_matrix)
plt.matshow(c_matrix,cmap=plt.cm.Blues)#,
for i in range(c_matrix.shape[0]):
    for j in range(c_matrix.shape[1]):
        plt.text(x=j, y=i, s=c_matrix[i, j])
plt.savefig(pca_path+str(versus_component)+'knn_comfusion_matrix'+'.png')


correct=0
for i in range(len(predict_class)):
    if(predicted[i]==predict_class[i]):
        correct=correct+1
print('Test  Accuracy:%.3f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))
print('-2 vs -3 :%.3f'%((c_matrix[0,0]+c_matrix[1,1])/(c_matrix[0,0]+c_matrix[0,1]+c_matrix[1,0]+c_matrix[1,1])),
      '({}/{})'.format((c_matrix[0,0]+c_matrix[1,1]),(c_matrix[0,0]+c_matrix[0,1]+c_matrix[1,0]+c_matrix[1,1])))
# fuzzy_point=0.5
# correct=0
# for i in range(len(predict_class)):
#     if(predicted[i]==predict_class[i]):
#         correct=correct+1
#     elif(abs(predicted[i]-predict_class[i])==1):
#         correct=correct+fuzzy_point

# # print('Test  Accuracy(wide +-1):%.3f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))


# for i in range(len(predict_class)):
#     if(int(predict_class[i])<=-2):
#         predict_class2[i]=-2
#     if(int(predicted[i])<=-2):
#         predicted[i]=-2

# correct=0
# for i in range(len(predict_class2)):
#     if(predicted[i]==predict_class2[i]):
#         correct=correct+1
    

# print('Test  Accuracy(only ripe or not)(no+-1):%.3f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))




# print('\nafter voting')
# predicted3=clf2.predict(new)
# new = test_data[versus_component].values
# predict_class = test_data['ripness'].values
# #引入voting
# ids = test_data['id'].values
# predicted2=classification_voting(predicted3,ids)
# # print(len(predicted2))
# d_matrix = metrics.confusion_matrix(predict_class,predicted2)
# print(d_matrix)
# print(predict_class[0:40])
# print(predict_class2[0:40])
# # print(test_reduced)
# # print(predict_class2[0:300])
# # print(predicted2[0:300])
# correct=0
# for i in range(len(predict_class)):
#     if(predicted2[i]==predict_class[i]):
#         correct=correct+1
# print('Test  Accuracy:%.2f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))
# #Test  Accuracy:0.70 14/20

# fuzzy_point=0.5
# correct=0
# for i in range(len(predict_class)):
#     if(predicted2[i]==predict_class[i]):
#         correct=correct+1
#     elif(abs(predicted2[i]-predict_class[i])==1):
#         correct=correct+fuzzy_point

# print('Test  Accuracy(wide +-1):%.2f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))


# # for i in range(len(predict_class2)):
# #     if(int(predict_class2[i])<=-2):
# #         predict_class[i]=-2
# #     if(int(predicted[i])<=-2):
# #         predicted2[i]=-2

# correct=0
# for i in range(len(predict_class2)):
#     if(predicted2[i]==predict_class2[i]):
#         correct=correct+1
    

# print('Test  Accuracy(only ripe or not)(no+-1):%.2f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))



# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(20, 15))
# ax = Axes3D(fig, elev=48, azim=134)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
#            cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)

# for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean(),
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)

# ax.set_title("3D visualization", fontsize=40)
# ax.set_xlabel("Sepal Length [cm]", fontsize=25)
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("Sepal Width [cm]", fontsize=25)
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("Petal Length [cm]", fontsize=25)
# ax.w_zaxis.set_ticklabels([])

# plt.show()