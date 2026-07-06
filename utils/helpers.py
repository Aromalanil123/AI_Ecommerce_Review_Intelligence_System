#Utility functions for text preprocessing, NLTK resource management, and advanced feature extraction for review analysis.
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import numpy as np
import pandas as pd

# Ensure resources are available
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesses text for NLP models."""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back
    return " ".join(words)

def extract_advanced_features(row):
    text = row['text_']
    rating = row['rating']
    if not isinstance(text, str):
        text = ""
        
    review_length = len(text.split())
    punctuation_count = sum([1 for char in text if char in string.punctuation])
    
    words = text.lower().split()
    unique_words = set(words)
    repeated_words_ratio = 1.0 - (len(unique_words) / max(len(words), 1))
    
    tb_polarity = TextBlob(text).sentiment.polarity
    
    # Map rating 1-5 to expected polarity -1 to 1
    rating_polarity_expected = (rating - 3) / 2.0
    mismatch_score = abs(tb_polarity - rating_polarity_expected)
    
    return pd.Series([review_length, punctuation_count, repeated_words_ratio, tb_polarity, mismatch_score])
