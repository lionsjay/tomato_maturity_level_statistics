# importing the libraries
#https://github.com/mehboobali98/IRIS-classification-using-SVMs/blob/main/Linear_SVM.ipynb
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import joblib #https://ithelp.ithome.com.tw/articles/10197575
from sklearn.model_selection import train_test_split
from voting import classification_voting
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import warnings
warnings.filterwarnings("ignore")
# loading iris data set
iris = datasets.load_iris()

dataset_name='19th2_tomato'
train_data=pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_train/merged.csv',delimiter=",")
test_data=pd.read_csv('F:\dataset/'+str(dataset_name)+'\svm_test/merged.csv',delimiter=",")
pca_path='F:\dataset/'+str(dataset_name)+'\pca/'
# versus_component=['red','green','blue']
# versus_component=['ndvi','gndvi','grri']
versus_component=['red','green','blue','ndvi','gndvi','grri']
# versus_component=['red','blue']
# versus_component=['r-g','hue']
# versus_component=['r-g','l-a']
# versus_component=['g-b','l-a']
# versus_component=['r-g','green','s-h','l-a']
# versus_component=['r-g','green','s-h','l-a','ndvi','ndre']
# versus_component=['r-g','green','s-h','l-a','ndvi']
# versus_component=['r-g','l-a']
# versus_component=['r-g','hue']
# versus_component=['r-g','g-b','s-h','l-a']
# versus_component=['r-g','hue','saturation','brightness']
# versus_component=['red','green','blue','ndvi']
# versus_component=['red','green','blue','ndre']
# versus_component=['hue','saturation','brightness','ndvi']
# versus_component=['hue','saturation','brightness','ndre']
# versus_component=['red','hue','r-g']
# versus_component=['ndvi','ndre']
# versus_component=['ndvi','gndvi','grri']
# versus_component=['r-g','green','s-h','l-a','ndvi','gndvi','grri']
# versus_component=['red','green','r-g','g-b','hue','saturation','brightness','s-h','l','a','b','l-a']
print(versus_component)
weight_path='F:\dataset/'+str(dataset_name)+'\weights/svm/'+str(versus_component) #儲存權重檔的位置

X =  train_data[versus_component].values
y = train_data['ripness'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)
degree=len(versus_component) # degree must be between 0 and min(n_samples, n_features)=3
# print(X.shape)

# """ 
# separating features and labels
# iris_x contains features
# iris_y contains labels 
# """
# iris_X = iris.data
# iris_y = iris.target
# # print(iris_X.shape)
# # print(iris_X)
# # print(iris_y.shape)
# # print(iris_y)

# # splitting the data set for training and testing parts
# indices = np.random.permutation(len(iris_X))

# X_train = iris_X[indices[:-20]]
# y_train = iris_y[indices[:-20]]
# X_test = iris_X[indices[-20:]]
# y_test = iris_y[indices[-20:]]

# # iris_X_train = iris_X[indices[:-20]]
# # iris_y_train = iris_y[indices[:-20]]
# # iris_X_test = iris_X[indices[-20:]]
# # iris_y_test = iris_y[indices[-20:]]
# # print(iris_X_train.shape)
# # print(iris_X_test.shape)

# creating folds list with k=5 and k = 10
k_values = [3,10]

# list to store different fold objects
folds = []
start=time.time()
for i in k_values:
    folds.append(KFold(n_splits = i, shuffle = True, random_state =101))

# specify range of C
c = np.arange(10,40,2)   #https://stackoverflow.com/questions/48373710/sklearn-model-selection-gridsearchcv-valueerror-c-0
              
# creating hyper-parameter dictionary
hyper_params = [{'C': c}]

# creating linear SVM model
model = svm.SVC(kernel="rbf")
              
models_cv = []

for i in range(2):
    # set up GridSearchCV()
    model_cv = GridSearchCV(estimator = model, param_grid = hyper_params, scoring= 'accuracy', cv = folds[i], verbose = 1,
                                                                                    return_train_score=True, n_jobs = -1)
    models_cv.append(model_cv)
    models_cv[i].fit(X_train,y_train)

# list to store cross validation results
cv_results = []

for i in range(len(k_values)):
    cv_r = pd.DataFrame(models_cv[i].cv_results_)
    cv_results.append(cv_r)


def best_params(models_cv, k_values):
    
    for i in range(len(k_values)):
        best_score = models_cv[i].best_score_
        best_hyperparams = models_cv[i].best_params_
        print("The best mean test score is {0} corresponding to hyperparameters {1} with k = {2}".format(best_score,best_hyperparams,k_values[i]))
best_params(models_cv,k_values)
# The best mean test score is 0.976923076923077 corresponding to hyperparameters {'C': 0.75} with k = 5
# The best mean test score is 0.9923076923076923 corresponding to hyperparameters {'C': 0.5} with k = 10

# Training the classifier using the best 'C' value and then testing on unseen data
# Using k=5 for k-Fold Cross Validation

best_hyperparams = models_cv[1].best_params_

print("The best C value: ", best_hyperparams['C'])



features_names = versus_component



# creating a linear SVM classifier
svc = svm.SVC(kernel= "rbf", C = best_hyperparams['C'],decision_function_shape='ovo')

# perform the training
linear_svm_3d=svc.fit(X_train,y_train)
end=time.time()
print('訓練時間:'+str(end-start)+'秒')


# prediction
y_pred = svc.predict(X_test)

# accuracy
print("Accuracy:", metrics.accuracy_score(y_true=y_test,y_pred=y_pred), "\n")

# printing the confusion matrix
c_matrix = metrics.confusion_matrix(y_test,y_pred)
print(c_matrix)

# import matplotlib.pyplot as plt
# import numpy as np

# mat = np.arange(1, 10).reshape(3, 3)

# # plt.matshow(mat, cmap=plt.cm.BrBG)
# for i in range(mat.shape[0]):
#     for j in range(mat.shape[1]):
#         plt.text(x=j, y=i, s=mat[i, j])
               
plt.show()

print("Evaluating on test data: ")
print(metrics.classification_report(y_test,y_pred))

# print("drawing feature importance")
# from sklearn.inspection import permutation_importance
# import matplotlib.pyplot as plt

# perm_importance = permutation_importance(svc, X_test, y_test)

# feature_names = versus_component
# features = np.array(feature_names)

# sorted_idx = perm_importance.importances_mean.argsort()
# print(features[sorted_idx])
# print(perm_importance.importances_mean[sorted_idx])
# plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
# plt.xlabel("Permutation Importance")
# plt.savefig(pca_path+str(versus_component)+'importance'+'.png')

# https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
# print("drawing feature importance")
# from sklearn.inspection import permutation_importance
# import matplotlib.pyplot as plt

# # X =  train_data[versus_component].values
# # y = train_data['ripness'].values
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

# perm_importance = permutation_importance(svc, new, predict_class)
# # perm_importance = permutation_importance(svc, X_test, y_test)

# feature_names = versus_component
# features = np.array(feature_names)

# sorted_idx = perm_importance.importances_mean.argsort()
# print(features[sorted_idx])
# print(perm_importance.importances_mean[sorted_idx])
# plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
# plt.xlabel("Permutation Importance")
# plt.savefig(pca_path+str(versus_component)+'importance'+'.png')


# joblib.dump(svc,'clf.pkl')#儲存權重檔
# clf2 = joblib.load('clf.pkl')
joblib.dump(svc,weight_path+'.pkl')#儲存權重檔
clf2 = joblib.load(weight_path+'.pkl')

#https://ithelp.ithome.com.tw/articles/10197575
new = test_data[versus_component].values
predict_class = test_data['ripness'].values
predict_class2 = test_data['ripness'].values
ids = test_data['id'].values
# print(predict_class)
# print(predict_class)
#[[231.9162695  118.2381781   87.32090492   7.29249437]
# [185.3584926  182.6752758  130.7640943   29.04894868]]

from sklearn.decomposition import PCA
# n_components=2 must be between 0 and min(n_samples, n_features)=1
pca2 = PCA(n_components=degree, iterated_power=1)
# pca = PCA(n_components=2, iterated_power=1)
test_reduced = pca2.fit_transform(new)
# predicted=clf2.predict(test_reduced)
predicted=clf2.predict(new)
# print(predicted[0:30])
predicted=classification_voting(predicted,ids)
print('voting')
# print(predicted[0:30])

c_matrix = metrics.confusion_matrix(predict_class,predicted)
print(c_matrix)
c_matrix.tolist()
# comfusion_matrix=np.array(c_matrix)
plt.matshow(c_matrix,cmap=plt.cm.Blues)#,
for i in range(c_matrix.shape[0]):
    for j in range(c_matrix.shape[1]):
        plt.text(x=j, y=i, s=c_matrix[i, j])
plt.savefig(pca_path+str(versus_component)+'svm_comfusion_matrix'+'.png')
plt.clf()

print("Evaluating on test data: ")
print(metrics.classification_report(y_test,y_pred))

# https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
print("drawing feature importance")
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# X =  train_data[versus_component].values
# y = train_data['ripness'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

perm_importance = permutation_importance(svc, new, predict_class)
# perm_importance = permutation_importance(svc, X_test, y_test)

feature_names = versus_component
features = np.array(feature_names)

sorted_idx = perm_importance.importances_mean.argsort()
print(features[sorted_idx])
print(perm_importance.importances_mean[sorted_idx])
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.savefig(pca_path+str(versus_component)+'importance'+'.png')


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

# print('Test  Accuracy(wide +-1):%.3f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))


# for i in range(len(predict_class)):
#     if(int(predict_class[i])<=-2):
#         predict_class[i]=-2
#     if(int(predicted[i])<=-2):
#         predicted[i]=-2

# correct=0
# for i in range(len(predict_class)):
#     if(predicted[i]==predict_class[i]):
#         correct=correct+1
    

# print('Test  Accuracy(only ripe or not)(no+-1):%.3f'%(correct/len(predict_class)),'({}/{})'.format(correct ,len(predict_class)))

# new = test_data[versus_component].values
# predicted3=clf2.predict(new)
# print('\nafter voting')
# #引入voting
# ids = test_data['id'].values
# predicted2=classification_voting(predicted3,ids)
# # print(len(predicted2))
# c_matrix = metrics.confusion_matrix(predict_class2,predicted2)
# print(c_matrix)
# # print(test_reduced)
# # print(predicted2[1443:1630])
# accuracy = svc.score(new, predict_class2)
# print('Test  Accuracy:%.2f'%accuracy,'({}/{})'.format(int(len(predicted2)*accuracy) ,len(predicted2)))
# #Test  Accuracy:0.70 14/20

# fuzzy_point=0.5
# correct=0
# for i in range(len(predict_class2)):
#     if(predicted2[i]==predict_class2[i]):
#         correct=correct+1
#     elif(abs(predicted2[i]-predict_class2[i])==1):
#         correct=correct+fuzzy_point

# print('Test  Accuracy(wide +-1):%.2f'%(correct/len(predict_class2)),'({}/{})'.format(correct ,len(predict_class2)))


# for i in range(len(predict_class2)):
#     if(int(predict_class2[i])<=-2):
#         predict_class2[i]=-2
#     if(int(predicted[i])<=-2):
#         predicted2[i]=-2

# correct=0
# for i in range(len(predict_class2)):
#     if(predicted2[i]==predict_class[i]):
#         correct=correct+1
    

# print('Test  Accuracy(only ripe or not)(no+-1):%.2f'%(correct/len(predict_class2)),'({}/{})'.format(correct ,len(predict_class2)))


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy 
# # drop the last two features for plotting

# iris_X_train = X_train[:,:2]
# iris_X_test = X_test[:,:2]
# iris_y_train =  y_train
# # # plotting decision boundary using gamma, C
# def plot_db(C, fig_size):
#     # svc = svm.SVC(kernel="linear", C=C)
#     # pred = svc.fit(iris_X_train, iris_y_train)
#     U, V = iris_X_train[:, 0], iris_X_train[:, 1]
#     xx, yy = make_meshgrid(U, V)
#     figsize = fig_size
#     fig = plt.figure(figsize=(figsize,figsize))
#     ax = plt.subplot(111)
#     plot_contours(ax, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#     ax.scatter(U, V, c=iris_y_train, cmap=plt.cm.coolwarm, s=20, edgecolors="k")

#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())

#     ax.set_xlabel('Sepal length')
#     ax.set_ylabel('Sepal width')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     title = "C= " + str(C)
#     ax.set_title(title)

#     plt.show()
# plot_db(best_hyperparams['C'], 8)

# X_train, X_test, y_train, y_test