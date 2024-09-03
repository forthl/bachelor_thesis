import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Spectral:

    def __init__(self, data, max_k=20, optimal_k_method='sil'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method

    def optimal_k_sil(self):
        ks = np.arange(2, self.max_k)
        silhouette_scores = []
        best_score = -1
        optimal_k = -1

        for k in ks:
            spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
            labels = spectral.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                optimal_k = k

        plt.plot(ks, silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs Number of Clusters")
        plt.show()

        return optimal_k

    def find_optimal_k(self):
        if self.optimal_k_method == 'sil':
            optimal_k = self.optimal_k_sil()
        else:
            optimal_k = 0
        return optimal_k

    def spectral_clustering(self, k):
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')

        # data points assigned to a cluster
        labels = spectral.fit_predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.spectral_clustering(optimal_k)
        return labels, centroids, self.data


class GaussianMixtureModel:

    def __init__(self, data, max_k=20, optimal_k_method='bic', n_init=5,
                 covariance_type='full', init_params='k-means++'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.init_params = init_params

    def optimal_k_elbow(self):
        # TODO
        return 1

    def optimal_k_bic(self):
        ks = np.arange(1, self.max_k)
        bics = []
        for k in ks:
            gmm = GaussianMixture(n_components=k, n_init=self.n_init, covariance_type=self.covariance_type,
                                  init_params=self.init_params, max_iter=1000)
            gmm.fit(self.data)
            bics.append(gmm.bic(self.data))

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(ks, bics)
        ax.set_xlabel(r'Number of clusters, $k$')
        ax.set_ylabel('BIC')
        ax.set_xticks(ks);

        diff = [x - bics[i - 1] for i, x in enumerate(bics)][1:]

        return diff.index(min(diff)) + 2

    def find_optimal_k(self):
        if self.optimal_k_method == 'bic':
            optimal_k = self.optimal_k_bic()
        else:
            optimal_k = 0
        return optimal_k

    def gmm_clustering(self, k):
        gmm = GaussianMixture(n_components=k, n_init=self.n_init, covariance_type=self.covariance_type,
                              init_params=self.init_params, max_iter=1000).fit(self.data)

        # data points assigned to a cluster
        labels = gmm.predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.gmm_clustering(optimal_k)
        return labels, centroids, self.data


class BayesianGaussianMixtureModel:
    def __init__(self, data, max_k=20, bgmm_weights_threshold=0.05, n_init=10,
                 covariance_type='full', init_params='random'):
        self.data = data
        self.max_k = max_k
        self.bgmm_weights_threshold = bgmm_weights_threshold
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.init_params = init_params

    def find_optimal_k(self):
        bgmm = BayesianGaussianMixture(n_components=self.max_k, covariance_type=self.covariance_type,
                                       init_params=self.init_params)
        bgmm.fit(self.data)
        bgmm_weights = bgmm.weights_
        optimal_k = (np.round(bgmm_weights, 2) > self.bgmm_weights_threshold).sum()

        return optimal_k

    def bgmm_clustering(self, k):
        bgmm = BayesianGaussianMixture(n_components=k, covariance_type=self.covariance_type,
                                       init_params=self.init_params, max_iter=2000).fit(self.data)

        # data points assigned to a cluster
        labels = bgmm.predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        #print(f"Optimal k: {optimal_k}")
        labels, centroids = self.bgmm_clustering(optimal_k)
        return labels


class Dbscan:
    def __init__(self, data, epsilon, min_samples):
        self.data = data
        self.epsilon = epsilon
        self.min_samples = min_samples

    def find_clusters(self):
        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
        labels = dbscan.fit_predict(self.data)

        return labels


class Optics:
    def __init__(self, data, min_samples, max_eps=None, metric='minkowski', cluster_method='xi'):
        self.data = data
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.cluster_method = cluster_method

    def find_clusters(self):
        optics = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps, metric=self.metric,
                        cluster_method=self.cluster_method)
        labels = optics.fit_predict(self.data)

        return labels


class Kmeans:
    def __init__(self, data, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.data = data
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def find_clusters(self):
        kmeans = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
                        max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
        labels = kmeans.fit_predict(self.data)

        return labels
