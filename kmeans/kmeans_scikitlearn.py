import numpy as np
class KMeans():
    def __init__(self, k, max_iteration = 10):
        self.k = k
        self.max_iteration = max_iteration
        self.all_label = []
        self.all_centroid = []
    def fit(self, datasets):
        #so chieu
        newFeatures = datasets.shape[1]
        #Khai bao so luong tam ngau nhien = ham get_random_centroid
        centroids = self.get_random_centroid(self.k, newFeatures)
        #them vao all_centroid, all_label
        self.all_centroid.append(centroids)
        self.all_label.append(None)
        #khoi tao bien iteration, oldCentroid
        iteration = 0
        OldCentroids = []

        #vong lap cap nhap centroid cho thuat toan
        while not self.should_stop(iteration, OldCentroids, centroids):
            iteration +=1
            OldCentroids = centroids
            labels= self.get_label(datasets, centroids)
            self.all_label.append(labels)

            centroids = self.get_centroid(datasets, labels, self.k)
            self.all_centroid.append(centroids)
            
        return centroids



    def get_random_centroid(self, newFeatures, k):
        return np.random.rand(k, newFeatures)

    def should_stop(self, iteration, OldCentroids, centroids):
        if(iteration > self.max_iteration):
            return True
        if OldCentroids is None or len(OldCentroids)==0:
            return False
        return np.all(OldCentroids == centroids)
    def get_label(self, datasets, centroids):
        labels = []
        for x in datasets:
            distance = np.sum((x-centroids)**2, axis=1)
            label = np.argmin(distance)
            labels.append(label)
        return labels
    def get_centroid(self, datasets, labels, k):
        centroids = []
        for i in np.arange(k):
            idx_i = np.where(np.array(labels)==i)[0]
            centroids_i = datasets[idx_i,:].mean(axis = 0)
            centroids.append(centroids_i)
        return np.array(centroids)
