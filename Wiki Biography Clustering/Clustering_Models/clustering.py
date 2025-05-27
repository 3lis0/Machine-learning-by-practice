from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np

def optimal_cluster_count(data):

    # Calculate inertia for different numbers of clusters
    inertia = []
    for k in range(1, 11):  # Test k values from 1 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Use the KneeLocator to find the elbow point
    knee = KneeLocator(range(1, 11), inertia, curve="convex", direction="decreasing")
    optimal_k = knee.elbow

    return optimal_k, inertia


def KMeans_clustring(data, n_clusters):
    
    # Create a KMeans instance with the specified number of clusters and random state
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    return kmeans

def hierarchical_clustering(data, n_clusters):
    """
    Perform hierarchical clustering on the given data."
    """

    # Perform hierarchical clustering
    linkage_matrix = linkage(data.toarray(), method='ward')

    # Get the optimal number of clusters

    labels = fcluster(linkage_matrix, n_clusters, criterion='distance')

    return linkage_matrix, labels

def gaussian_mixture_model(data, n_clusters):
    """
    Perform Gaussian Mixture Model clustering on the given data.
    """

    # Create a Gaussian Mixture Model instance with the specified number of clusters and random state
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)

    # Fit the model to the data
    gmm.fit(data.toarray())

    return gmm
