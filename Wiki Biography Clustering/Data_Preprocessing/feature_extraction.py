# Feature Extraction using TF-IDF
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

def tune_max_features(data, n_iter=5, random_state=42):
    """
    Tune max_features for TfidfVectorizer using RandomizedSearchCV without labels.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'text' column.
        n_iter (int): Number of random search iterations.
        random_state (int): Random seed.

    Returns:
        int: Best max_features value found.
    """
    # Define the pipeline (TF-IDF only, no classifier)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english'))
    ])

    # Define the hyperparameter search space
    param_dist = {
        'tfidf__max_features': np.random.randint(10, 500, 10),  # Random values between 10 and 500
    }

    # Define a scoring function (maximize variance of TF-IDF scores)
    def score_func(estimator, X):
        tfidf_matrix = estimator.named_steps['tfidf'].transform(data["text"])
        return np.var(tfidf_matrix.toarray())  # Use variance as a measure of informativeness

    # Run Randomized Search
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        n_iter=n_iter, scoring=score_func, verbose=1, random_state=random_state
    )

    # Fit the model
    random_search.fit(data["text"])

    # Return the best max_features value
    return random_search.best_params_["tfidf__max_features"]

def feature_extraction(data):
    """
    Extract features using TF-IDF and find top words with their scores.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'text' column.

    Returns:
        tuple: (TF-IDF matrix, feature names, dictionary of top words with scores)
    """
    # Get optimal max_features value
    best_max_features = tune_max_features(data)
    
    # Apply TF-IDF with optimal max_features
    vectorizer = TfidfVectorizer(max_features=best_max_features, stop_words="english")
    X_tfidf = vectorizer.fit_transform(data["text"])
    
    # Extract feature names
    feature_names = vectorizer.get_feature_names_out()

    # Compute mean TF-IDF scores for each word
    tfidf_scores = np.mean(X_tfidf.toarray(), axis=0)

    # Sort words by TF-IDF scores
    top_words_indices = np.argsort(tfidf_scores)[::-1]
    top_words_dict = {feature_names[i]: tfidf_scores[i] for i in top_words_indices}

    return X_tfidf, feature_names, top_words_dict
