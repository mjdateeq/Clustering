import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


## Random Dataset
X = [4, 5, 6 , 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21, 16]

data =  list(zip(X,y))
inertias = []

# elbow method

for i in range (2, 10):
    kmeans = KMeans (n_clusters=i, init="random")
##    kmeans = KMeans (n_clusters=i)

    kmeans.fit (data)
    inertias.append(kmeans.inertia_)


cen = list(range (2,10)) ## for plotting purposes

plt.plot( cen, inertias)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show() 
