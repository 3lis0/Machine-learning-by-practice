from sklearn.metrics import silhouette_score

def silhouette_metric(data, labels):
    """
    Calculate the silhouette score for the given data and labels.

    Parameters:
    data (array-like): Feature data.
    labels (array-like): Cluster labels.

    Returns:
    float: Silhouette score.
    """
    return silhouette_score(data, labels)