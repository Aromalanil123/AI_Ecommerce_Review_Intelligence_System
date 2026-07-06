"""Trains a sentiment classification model using Logistic Regression and TF-IDF features to classify reviews as Positive, Neutral, or Negative."""

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from utils.helpers import preprocess_text

def map_sentiment(rating):
    if rating > 3:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

def train_sentiment():
    print('Starting sentiment model training...')
    
    csv_path = os.path.join(os.getcwd(), 'dataset', 'fake_reviews_dataset.csv')
    df = pd.read_csv(csv_path)
    
    # Preprocessing
    df['sentiment'] = df['rating'].apply(map_sentiment)
    print('Preprocessing text...')
    df['cleaned_review'] = df['text_'].apply(preprocess_text)
    
    X = df['cleaned_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    print('Training model (Logistic Regression)...')
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    print('Evaluating model...')
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/sentiment_pipeline.pkl')
    print('Sentiment pipeline saved to models/sentiment_pipeline.pkl')

if __name__ == "__main__":
    train_sentiment()
