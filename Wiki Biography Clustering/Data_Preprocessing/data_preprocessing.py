import spacy
import pandas as pd
import re

# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Process text using spaCy
    doc = nlp(text)

    # Tokenization, Stop-word removal, and Lemmatization
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Join tokens back into a cleaned text string
    return " ".join(clean_tokens)