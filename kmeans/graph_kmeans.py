import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from kmeans_scikitlearn import KMeans

datasets, _ = make_blobs(n_samples = 100, n_features = 2, cluster_std =3.0, random_state = 123)
kmeans = KMeans(k = 2, max_iteration = 9)
centroids = kmeans.fit(datasets)

# print(datasets)

# ve graph
gs = GridSpec(nrows = 3, ncols = 3)
plt.figure(figsize = (15,15), label = "kmeans")

color = ['blue', 'green']
labels = ['c1', 'c2']
for i in np.arange(len(kmeans.all_centroid)):
    plt.subplot(gs[i])
    #ve hinh dau tien ch co gi thay doi
    if i ==0:
        centroids_i = kmeans.all_centroid[i]
        plt.scatter(datasets[:,0], datasets[:,1], s=40, alpha =0.5,color = 'red')
        # ve tam
        for j in np.arange(kmeans.k):
            plt.scatter(centroids[j,0], centroids[j,1], marker = "x", s= 80, color = "red")
        plt.title("ORIGINAL DATASET")
    else:
        #lay centroid va label 
        centroids_i = kmeans.all_centroid[i]
        labels_i = kmeans.all_label[i]
        for j in np.arange(kmeans.k):
            idx_j = np.where(np.array(labels_i) == j)[0]
            plt.scatter(datasets[idx_j, 0], datasets[idx_j, 1], color=color[j], label=labels[j], s=40, alpha=0.3, lw = 0)
            plt.scatter(centroids_i[j, 0], centroids_i[j, 1], marker='x', color=color[j], s=80, label=labels[j])
        plt.title(f"iteration {i}")
plt.tight_layout()
plt.show()