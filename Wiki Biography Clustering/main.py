import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from Data.load_data import load_wikipedia_data
from Data_Preprocessing.data_preprocessing import clean_text
from Data_Preprocessing.feature_extraction import feature_extraction
from Clustering_Models.clustering import optimal_cluster_count, KMeans_clustring, hierarchical_clustering, gaussian_mixture_model
from Model_evaluation.model_evaluate import silhouette_metric
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

# Ensure the project root is in sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

if __name__ == '__main__':
    # Load data
    data_path = "Wiki_Biography_Clustering_Project\Data\people_wiki.csv"
    data = load_wikipedia_data(data_path)
    print("Data Loaded Successfully")
    
    # Preprocess text
    data.loc[:, "text"] = data["text"].apply(clean_text)
    print("Text Preprocessing Completed")
    
    # Feature Extraction
    tfidf_matrix = feature_extraction(data)[0]
    print("Feature Extraction Done")
    
    # Find Optimal Clusters
    optimal_k = optimal_cluster_count(tfidf_matrix)[0]
    print(f"Optimal Cluster Count: {optimal_k}")
    
    # Apply Clustering Algorithms
    kmeans_model = KMeans_clustring(tfidf_matrix, optimal_k)
    kmeans_labels = kmeans_model.labels_
    hierarchical, hierarchical_labels = hierarchical_clustering(tfidf_matrix, optimal_k)
    gmm_model = gaussian_mixture_model(tfidf_matrix, optimal_k)
    gmm_labels = gmm_model.predict(tfidf_matrix.toarray())
    
    # Evaluate Clustering Performance
    kmeans_score = silhouette_metric(tfidf_matrix, kmeans_labels)
    hierarchical_score = silhouette_metric(tfidf_matrix, hierarchical_labels)
    gmm_score = silhouette_metric(tfidf_matrix, gmm_labels)
    
    print(f"Silhouette Scores:\n KMeans: {kmeans_score}\n Hierarchical: {hierarchical_score}\n GMM: {gmm_score}")
    
    # Save Models
    print("Saving trained models...")
    joblib.dump(kmeans_model, "clustering_model.pkl")
    joblib.dump(tfidf_matrix, "vectorizer.pkl")
    print("Models saved successfully!\n")

    # Visualize PCA Projection
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(10,6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Cluster')
    plt.title("PCA Projection of Clustering")
    plt.show()
    
    print("Clustering and Evaluation Completed!")
