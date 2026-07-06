"""Trains a fake review detection model using Random Forest with TF-IDF text features and engineered numerical features to identify AI-generated or manipulated reviews."""

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from utils.helpers import preprocess_text, extract_advanced_features

def train_fake_review():
    print('Starting fake review model training...')
    
    csv_path = os.path.join(os.getcwd(), 'dataset', 'fake_reviews_dataset.csv')
    df = pd.read_csv(csv_path)
    
    print('Extracting advanced features...')
    df[['review_length', 'punctuation_count', 'repeated_words_ratio', 'tb_polarity', 'mismatch_score']] = df.apply(extract_advanced_features, axis=1)
    
    print('Preprocessing text...')
    df['cleaned_review'] = df['text_'].apply(preprocess_text)
    
    X = df[['cleaned_review', 'review_length', 'punctuation_count', 'repeated_words_ratio', 'tb_polarity', 'mismatch_score']]
    y = df['label'] # CG or OR
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=3000), 'cleaned_review'),
            ('num', StandardScaler(), ['review_length', 'punctuation_count', 'repeated_words_ratio', 'tb_polarity', 'mismatch_score'])
        ])
    
    # Create Pipeline with Random Forest
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print('Training model (Random Forest)...')
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    print('Evaluating model...')
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fake_review_pipeline.pkl')
    print('Fake review pipeline saved to models/fake_review_pipeline.pkl')

if __name__ == "__main__":
    train_fake_review()
