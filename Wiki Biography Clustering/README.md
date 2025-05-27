# **Wikipedia Biography Clustering Project**

## Overview
This project applies unsupervised learning techniques to cluster documents from the **People Wikipedia Dataset**. The goal is to uncover inherent structures within the dataset, grouping biographies of notable individuals based on textual similarities.

## Datasets

### 1. People Wikipedia Dataset
**Description:**
The **People Wiki Dataset** consists of biographical articles of notable individuals extracted from Wikipedia. Each entry contains **a unique URI, the person's name, and text extracted from their Wikipedia page**.

**Features:**
- **URI**: A unique identifier for each person’s Wikipedia page.
- **Name**: Full name of the individual.
- **Text**: Extracted content from their Wikipedia biography.

## Methodology

The project follows these key steps:

1. **Data Collection**: Loading and preparing the dataset.
2. **Data Preprocessing**:
   - Lowercasing
   - Removing punctuation and special characters
   - Stopword removal
   - Lemmatization
   - Tokenization
3. **Feature Extraction**:
   - **TF-IDF Vectorization**: Converting text into numerical representations.
4. **Clustering Algorithms**:
   - **K-Means Clustering**
   - **Hierarchical Clustering**
   - **Gaussian Mixture Model (GMM)**
5. **Evaluation Metrics**:
   - **Silhouette Score**
6. **Visualization**:
   - **t-SNE & PCA**: Dimensionality reduction for visual cluster representation.
   - **Dendrograms**: Visualizing hierarchical clustering.

## Folder Structure

```
├── README.md
├── main.py                      # Main script to run the entire pipeline
├── requirements.txt             # List of required Python libraries
├── app/
│   ├── app.py                   # Deployment script for web app
│   ├── kmeans_model.pkl          # Saved KMeans clustering model
│   ├── vectorizer.pkl            # Saved TF-IDF vectorizer
├── cluster_models/
│   ├── clustering.py             # Implementation of clustering algorithms
├── data/
│   ├── load_data.py              # Script to load and prepare dataset
│   ├── people_wiki.csv           # Wikipedia dataset
├── data_preprocessing/
│   ├── data_preprocessing.py     # Text cleaning and preprocessing
│   ├── feature_extraction.py     # TF-IDF feature extraction
├── model_evaluation/
│   ├── model_evaluate.py         # Clustering evaluation metrics
├── visualization/
│   ├── notebook.ipynb                # Jupyter notebook for EDA & visualization
│   ├── people_wiki_subset.ipynb       # Notebook focusing on a subset of the dataset
```

## Installation & Usage

### Install Dependencies
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### Run the Pipeline
Execute the main script to perform clustering:
```sh
python main.py
```

### Deployment
To deploy the web app:
```sh
cd app
python app.py
```

## Tools & Technologies
- **Programming Language**: Python
- **Libraries**:
  - **Data Manipulation**: pandas, NumPy
  - **Text Processing**: NLTK, spaCy
  - **Machine Learning**: scikit-learn
  - **Visualization**: matplotlib, seaborn
- **Deployment**: Flask or Streamlit (for web interface)

## References
- **Wikipedia People Dataset**: Extracted structured content from Wikipedia. [Dataset Link](https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data)