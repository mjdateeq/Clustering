import numpy as np
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt


x = [0, 0.4, 0.3, 0.67, 0.87, 0.97, 1.2, 2, 2.4, 2.3, 2.67, 2.87, 2.97, 3.2]
y = [0.2, 0.67, 0.7, 0.45, 0.95, 1.05, 1.1, 2.5, 2.67, 2.7, 2.45, 2.95, 3.05, 3.1]

data =  list(zip(x,y))
num_data=len(data)
num_feature=np.ndim((np.array(data)))
k=2
init_weight = np.random.rand(num_data, num_feature)

def Def_Centroid (data, weight, num_cen, num_feature):
    centroid=np.zeros((num_cen,num_feature))

    for i in range (num_cen):
        for j in range (num_feature):
            Val1=np.sum(np.square(weight[:,i]))
            Val2=np.sum(np.square(init_weight[:,i])*data[:,j])
            centroid[i,j]=Val2/Val1

    return centroid

def UpdateWeight (arr, init_weight, cen):
    weight=np.zeros((init_weight.shape[0],init_weight.shape[1]))
    for i in range (init_weight.shape[0]):
        val = 0
        for j in range (init_weight.shape[1]):
            val = val + (1/np.linalg.norm(np.square((cen[j,:])-(arr[i,:]))))

        for j in range (init_weight.shape[1]):
            w=(1/np.linalg.norm(np.square((cen[j,:])-(arr[i,:]))))/val
            weight [i][j]= w

    return weight


def ObjFunc (weight, cen, arr):
    for j in range (init_weight.shape[1]):
        for i in range (init_weight.shape[0]):
            ObjVal= np.square(weight [i,j]) * np.square((np.linalg.norm(cen[j,:]-arr[i,:])))
    return ObjVal



arr = np.array (data)

for i in range (100):
    centroid = Def_Centroid (arr, init_weight, k, num_feature)
    init_weight = UpdateWeight (arr, init_weight, centroid)
    OptimizedValue = ObjFunc(init_weight, centroid, arr)


labels = pairwise_distances_argmin(arr, centroid)



plt.scatter(x, y, c=labels, s=50, cmap='viridis')
plt.scatter(centroid[0][0],centroid[0][1],s=50, marker='x')
plt.scatter(centroid[1][0],centroid[1][0],s=50, marker='x')

plt.show()

# credit to Tan Pham - pntan.iac@gmail.com

