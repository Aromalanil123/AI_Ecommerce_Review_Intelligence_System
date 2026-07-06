"""Orchestrator script that trains both sentiment classification and fake review detection models sequentially."""

import os
from sentiment_model import train_sentiment
from fake_review_model import train_fake_review

if __name__ == "__main__":
    print("========================================")
    print("=== Training Sentiment Analysis Model ===")
    print("========================================")
    train_sentiment()
    
    print("\n========================================")
    print("=== Training Fake Review Detection Model ===")
    print("========================================")
    train_fake_review()
    print("\n=== All Models Trained Successfully! ===")
