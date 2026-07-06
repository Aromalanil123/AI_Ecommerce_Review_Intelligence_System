# AI-Based E-commerce Review Intelligence System

A complete machine learning solution for e-commerce review analysis, designed to identify sentiment, detect fake reviews, and surface product insights through interactive analytics.

## 🚀 What the Project Does
This system analyzes customer reviews and provides:
- **Sentiment Analysis**: classifies reviews as *Positive*, *Neutral*, or *Negative*.
- **Fake Review Detection**: identifies reviews that may be generated or manipulated.
- **Product Insights**: summarizes trending topics, customer complaints, and category-level performance.
- **Data Visualization**: interactive charts, word clouds, and summary KPIs.

## 🔍 Key Features
- Real-time single review analysis with sentiment, authenticity, and polarity score
- Dashboard for sentiment distribution, fake review detection, rating spread, and category-level patterns
- AI-generated summary insights for product categories
- End-to-end model training using Scikit-Learn pipelines
- Support for feature engineering from review text and rating metadata

## 🧠 Architecture
- `app.py`: Streamlit dashboard for analysis and visualization
- `train_model.py`: orchestrates training for both models
- `sentiment_model.py`: trains sentiment classification pipeline
- `fake_review_model.py`: trains fake review detection pipeline
- `utils/helpers.py`: text preprocessing and feature extraction
- `dataset/fake_reviews_dataset.csv`: raw review dataset
- `models/`: serialized trained model pipelines

## 🛠️ Tech Stack
- Python, Streamlit
- Scikit-Learn, Pandas, NumPy
- NLTK, TextBlob
- Plotly, WordCloud, Matplotlib
- Joblib for model persistence

## 📁 Folder Structure
```text
AI_Ecommerce_Review_Intelligence_System/
├── app.py
├── fake_review_model.py
├── sentiment_model.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── fake_reviews_dataset.csv
├── models/
├── static/
├── templates/
└── utils/
    └── helpers.py
```

## ✅ Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the models:
   ```bash
   python train_model.py
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## 📌 Notes
- The dataset is sampled for dashboard performance.
- `train_model.py` saves pipelines to `models/sentiment_pipeline.pkl` and `models/fake_review_pipeline.pkl`.
- If the models are missing, the app prompts you to train them.

## 💡 Suggested Improvements
- Add more real-world review datasets for stronger generalization.
- Extend the fake review detector with deep learning or metadata features.
- Deploy as a microservice for higher availability and concurrency.

---

## 🧪 Model Training Details
- Sentiment model: Logistic Regression with TF-IDF text features
- Fake review detector: Random Forest using TF-IDF and engineered numerical features

## 📚 Interview / Project Talking Points
- The application demonstrates **end-to-end NLP**, from preprocessing through inference.
- It uses a combined **text + metadata** approach to detect fake reviews.
- Visual analytics help stakeholders identify both sentiment trends and suspicious review patterns.
