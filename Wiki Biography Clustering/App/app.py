import streamlit as st
import joblib
from Data_Preprocessing.data_preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the clustering model & vectorizer
kmeans_model = joblib.load("Wiki_Biography_Clustering_Project\App\kmeans_model.pkl")
tfidf_vectorizer = joblib.load("Wiki_Biography_Clustering_Project\App\Vectorizer.pkl")

# Streamlit UI
st.title("Wikipedia Biography Clustering")
st.write("Enter a biography text, and we'll predict its cluster.")

# Text input
user_input = st.text_area("Enter Biography Text:", "")

if st.button("Predict Cluster"):
    if user_input:
        # Preprocess and vectorize text
        clean_text_input = clean_text(user_input)

        vectorizer = TfidfVectorizer(vocabulary=tfidf_vectorizer)
        vectorizer.fit([clean_text_input])
        text_vector = vectorizer.transform([clean_text_input])

        # Predict cluster
        predicted_cluster = kmeans_model.predict(text_vector)[0]

        # Show result
        st.success(f"Predicted Cluster: {predicted_cluster}")
    else:
        st.warning("Please enter some text to predict.")

