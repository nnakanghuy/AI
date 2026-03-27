import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
img = plt.imread("img.jpg")
# print(img.shape)
height = img.shape[0]
width = img.shape[1]
img = img.reshape(width*height,3)
# print(img.shape)
kmeans = KMeans(n_clusters=4).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
# print(clusters)
img2 = np.zeros_like(img)
for i in range(len(img2)):
    img2[i] = clusters[labels[i]]
img2 = img2.reshape(height, width,3)
plt.imshow(img2)
plt.show()