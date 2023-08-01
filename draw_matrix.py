import matplotlib.pyplot as plt
import numpy as np

# mat = np.arange(1, 10).reshape(3, 3)
mat=np.array([[1,2],[3,4]])
plt.matshow(mat,cmap=plt.cm.Blues)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(x=j, y=i, s=mat[i, j])
               
plt.show()