"""Model loader that trains models if they don't exist."""

import os
import joblib

def ensure_models_exist():
    """Check if models exist, train them if they don't."""
    models_dir = os.path.join(os.getcwd(), 'models')
    sent_path = os.path.join(models_dir, 'sentiment_pipeline.pkl')
    fake_path = os.path.join(models_dir, 'fake_review_pipeline.pkl')
    
    models_exist = os.path.exists(sent_path) and os.path.exists(fake_path)
    
    if not models_exist:
        print("\n" + "="*50)
        print("Models not found. Training models...")
        print("="*50 + "\n")
        
        try:
            from sentiment_model import train_sentiment
            from fake_review_model import train_fake_review
            
            train_sentiment()
            print("\n")
            train_fake_review()
            
            print("\n" + "="*50)
            print("Models trained successfully!")
            print("="*50 + "\n")
        except Exception as e:
            print(f"Error training models: {e}")
            raise

def load_models():
    """Load models after ensuring they exist."""
    ensure_models_exist()
    
    base = os.path.join(os.getcwd(), 'models')
    sent_path = os.path.join(base, 'sentiment_pipeline.pkl')
    fake_path = os.path.join(base, 'fake_review_pipeline.pkl')
    
    try:
        sent = joblib.load(sent_path)
        fake = joblib.load(fake_path)
        return sent, fake
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        return None, None
