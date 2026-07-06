"""Streamlit web application for real-time e-commerce review analysis with sentiment classification and fake review detection."""

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from utils.helpers import preprocess_text, extract_advanced_features
from textblob import TextBlob

# ──────────────────────────────────────────────
# Page Config & CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Review Intelligence System",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(0,0,0,0.3); }
.metric-card h2 { color: #a78bfa; font-size: 2.2rem; margin: 0; font-weight: 700; }
.metric-card p  { color: #e2e8f0; font-size: 0.9rem; margin: 6px 0 0 0; font-weight: 400; }

/* Section headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #c4b5fd;
    margin: 32px 0 14px 0;
    border-left: 4px solid #7c3aed;
    padding-left: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #2d1b69 100%);
}
[data-testid="stSidebar"] * { color: #e0d7ff !important; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.3s;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca);
    box-shadow: 0 6px 20px rgba(124,58,237,0.5);
    transform: translateY(-2px);
}

/* Sentiment tags */
.tag-positive { background:#065f46; color:#6ee7b7; padding:6px 14px; border-radius:20px; font-weight:600; font-size:1rem; }
.tag-neutral  { background:#1e3a5f; color:#93c5fd; padding:6px 14px; border-radius:20px; font-weight:600; font-size:1rem; }
.tag-negative { background:#7f1d1d; color:#fca5a5; padding:6px 14px; border-radius:20px; font-weight:600; font-size:1rem; }
.tag-genuine  { background:#064e3b; color:#6ee7b7; padding:6px 14px; border-radius:20px; font-weight:600; font-size:1rem; }
.tag-fake     { background:#7c2d12; color:#fdba74; padding:6px 14px; border-radius:20px; font-weight:600; font-size:1rem; }

/* Insight boxes */
.insight-box {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Text area */
textarea { background: rgba(255,255,255,0.06) !important; color: #e2e8f0 !important; border: 1px solid rgba(167,139,250,0.3) !important; border-radius: 10px !important; }

/* General text */
h1, h2, h3 { color: #e2e8f0 !important; }
p, label, .stMarkdown { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
def map_authenticity_label(code):
    return 'Genuine' if code == 'OR' else 'Possibly Fake'

@st.cache_resource
def load_models():
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

# ──────────────────────────────────────────────
# Load & preprocess dataset (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_dashboard_data():
    path = os.path.join(os.getcwd(), 'dataset', 'fake_reviews_dataset.csv')
    df = pd.read_csv(path)
    # Sample for speed
    df = df.sample(2000, random_state=42).reset_index(drop=True)
    df['cleaned_review'] = df['text_'].apply(preprocess_text)

    # Sentiment label from rating
    def map_sent(r):
        if r > 3: return 'Positive'
        elif r == 3: return 'Neutral'
        return 'Negative'
    df['Sentiment'] = df['rating'].apply(map_sent)

    # Clean category name
    df['category_clean'] = df['category'].str.replace('_5', '').str.replace('_', ' ')

    return df

@st.cache_data
def run_predictions(_sent_pipe, _fake_pipe, df):
    """Run model predictions on the sampled dataset."""
    features = df.apply(extract_advanced_features, axis=1)
    features.columns = ['review_length', 'punctuation_count', 'repeated_words_ratio', 'tb_polarity', 'mismatch_score']
    df2 = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df2['Predicted_Sentiment'] = _sent_pipe.predict(df2['cleaned_review'])
    df2['Predicted_Authenticity'] = _fake_pipe.predict(df2)
    return df2

sent_pipe, fake_pipe = load_models()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.markdown("## 🛍️ AI Review Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🔍 Single Review Analysis",
    "📤 Upload & Analyze Dataset"
])
st.sidebar.markdown("---")


# ══════════════════════════════════════════════
# PAGE 1: Single Review Analysis
# ══════════════════════════════════════════════
if page == "🔍 Single Review Analysis":
    st.markdown("# 🔍 Single Review Analysis")
    st.markdown("Analyze any e-commerce review for **Sentiment** and **Authenticity** in real time.")
    st.markdown("---")

    col_in, col_ex = st.columns([2, 1])

    with col_in:
        review_text = st.text_area("📝 Paste your review here:", height=160,
            placeholder="e.g., Amazing product! Fast delivery and great quality. Highly recommend!")
        rating = st.slider("⭐ Product Rating", 1, 5, 4)
        analyze_btn = st.button("🚀 Analyze Review")

    with col_ex:
        st.markdown("**💡 Try these examples:**")
        st.info("\"Amazing product and fast delivery! Love it!\" → Positive")
        st.error("\"Best product ever!!!! Buy now!!!\" → Possibly Fake")
        st.warning("\"It's okay, nothing special.\" → Neutral")

    if analyze_btn:
        if not review_text.strip():
            st.warning("Please enter a review first.")
        elif not sent_pipe or not fake_pipe:
            st.error("Models not found. Please run `python train_model.py` first.")
        else:
            with st.spinner("Analyzing your review..."):
                # Sentiment prediction
                cleaned = preprocess_text(review_text)
                sentiment = sent_pipe.predict([cleaned])[0]

                # Fake prediction
                row_df = pd.DataFrame([{'text_': review_text, 'rating': rating}])
                feats = row_df.apply(extract_advanced_features, axis=1)
                feats.columns = ['review_length', 'punctuation_count', 'repeated_words_ratio', 'tb_polarity', 'mismatch_score']
                row_df = pd.concat([row_df, feats], axis=1)
                row_df['cleaned_review'] = cleaned
                authenticity = fake_pipe.predict(row_df)[0]
                auth_label = map_authenticity_label(authenticity)

                # Polarity score
                polarity = TextBlob(review_text).sentiment.polarity

            st.markdown("---")
            st.markdown("### 📋 Analysis Results")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**🎭 Sentiment**")
                if sentiment == 'Positive':
                    st.markdown('<span class="tag-positive">😊 Positive</span>', unsafe_allow_html=True)
                elif sentiment == 'Neutral':
                    st.markdown('<span class="tag-neutral">😐 Neutral</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-negative">😠 Negative</span>', unsafe_allow_html=True)

            with c2:
                st.markdown("**🕵️ Authenticity**")
                if auth_label == 'Genuine':
                    st.markdown('<span class="tag-genuine">✅ Genuine Review</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-fake">🤖 Possibly Fake</span>', unsafe_allow_html=True)

            with c3:
                st.markdown("**📈 Polarity Score**")
                color = "#6ee7b7" if polarity > 0 else ("#fca5a5" if polarity < 0 else "#93c5fd")
                st.markdown(f'<h2 style="color:{color}; margin:0">{polarity:.2f}</h2>', unsafe_allow_html=True)
                st.caption("Range: -1.0 (Very Negative) to +1.0 (Very Positive)")

            # Feature breakdown
            st.markdown("---")
            st.markdown("### 🔬 Feature Breakdown")
            fa1, fa2, fa3, fa4 = st.columns(4)
            fa1.metric("Word Count", len(review_text.split()))
            fa2.metric("Punctuation Count", sum(1 for c in review_text if c in '!?,;:'))
            fa3.metric("Rating Given", f"{rating} ⭐")
            fa4.metric("Polarity", f"{polarity:.2f}")

# ══════════════════════════════════════════════
# PAGE 3: Upload & Analyze Dataset
# ══════════════════════════════════════════════
elif page == "📤 Upload & Analyze Dataset":
    st.markdown("# 📤 Upload & Analyze Your Own Dataset")
    st.markdown("Upload a CSV file with reviews and get instant analytics powered by AI models.")
    st.markdown("---")

    if not sent_pipe or not fake_pipe:
        st.error("Models not found. Please run `python train_model.py` first.")
        st.stop()

    st.markdown("### 📋 Upload any CSV dataset")
    st.markdown("""
    Upload a CSV file and then map the columns to the review text, rating, and category fields.
    - **Review text** is required.
    - **Rating** is optional; missing values default to neutral (3).
    - **Category** is optional; missing values default to General.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing your dataset..."):
                df_upload = pd.read_csv(uploaded_file)

                if df_upload.empty:
                    st.error("❌ Uploaded CSV is empty.")
                    st.stop()

                columns = list(df_upload.columns)
                default_text = 'text_' if 'text_' in columns else columns[0]
                default_rating = 'rating' if 'rating' in columns else 'None'
                default_category = 'category' if 'category' in columns else 'None'

                text_column = st.selectbox("Select the review text column", columns, index=columns.index(default_text))
                rating_column = st.selectbox(
                    "Select the rating column (optional)",
                    ['None'] + columns,
                    index=(columns.index(default_rating) + 1) if default_rating in columns else 0
                )
                category_column = st.selectbox(
                    "Select the category column (optional)",
                    ['None'] + columns,
                    index=(columns.index(default_category) + 1) if default_category in columns else 0
                )

                df_upload['text_'] = df_upload[text_column].fillna("").astype(str)
                df_upload = df_upload[df_upload['text_'].str.strip() != ""]
                if df_upload.empty:
                    st.error("❌ No non-empty review text values were found in the selected text column.")
                    st.stop()

                if rating_column == 'None':
                    df_upload['rating'] = 3
                else:
                    df_upload['rating'] = pd.to_numeric(df_upload[rating_column], errors='coerce').fillna(3)

                if category_column == 'None':
                    df_upload['category'] = 'General'
                else:
                    df_upload['category'] = df_upload[category_column].fillna('General').astype(str)

                df_upload['cleaned_review'] = df_upload['text_'].apply(preprocess_text)

                def map_sent(r):
                    if r > 3: return 'Positive'
                    elif r == 3: return 'Neutral'
                    return 'Negative'
                df_upload['Sentiment'] = df_upload['rating'].apply(map_sent)

                df_upload['category_clean'] = df_upload['category'].astype(str).str.replace('_5', '').str.replace('_', ' ')

                # Run predictions
                df_upload = run_predictions(sent_pipe, fake_pipe, df_upload)

                st.success(f"✅ Loaded {len(df_upload)} reviews successfully!")
                st.markdown("---")

                # KPI Row
                total = len(df_upload)
                pos_pct = (df_upload['Predicted_Sentiment'] == 'Positive').mean() * 100
                fake_pct = (df_upload['Predicted_Authenticity'] == 'CG').mean() * 100
                avg_rating = df_upload['rating'].mean()

                k1, k2, k3, k4 = st.columns(4)
                k1.markdown(f'<div class="metric-card"><h2>{total}</h2><p>Total Reviews</p></div>', unsafe_allow_html=True)
                k2.markdown(f'<div class="metric-card"><h2>{pos_pct:.1f}%</h2><p>Positive Sentiment</p></div>', unsafe_allow_html=True)
                k3.markdown(f'<div class="metric-card"><h2>{fake_pct:.1f}%</h2><p>Fake Reviews</p></div>', unsafe_allow_html=True)
                k4.markdown(f'<div class="metric-card"><h2>{avg_rating:.2f}⭐</h2><p>Average Rating</p></div>', unsafe_allow_html=True)

                st.markdown("---")

                # Visualizations
                r1c1, r1c2 = st.columns(2)

                with r1c1:
                    st.markdown('<div class="section-title">🎭 Sentiment Distribution</div>', unsafe_allow_html=True)
                    sent_counts = df_upload['Predicted_Sentiment'].value_counts().reset_index()
                    fig_sent = px.pie(sent_counts, names='Predicted_Sentiment', values='count',
                                      color='Predicted_Sentiment',
                                      color_discrete_map={'Positive':'#6ee7b7','Neutral':'#93c5fd','Negative':'#fca5a5'},
                                      hole=0.45)
                    fig_sent.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                            font_color='#e2e8f0', legend_font_color='#e2e8f0', margin=dict(t=20))
                    st.plotly_chart(fig_sent, use_container_width=True)

                with r1c2:
                    st.markdown('<div class="section-title">🕵️ Genuine vs Fake Reviews</div>', unsafe_allow_html=True)
                    auth_counts = df_upload['Predicted_Authenticity'].value_counts().reset_index()
                    auth_counts['label'] = auth_counts['Predicted_Authenticity'].map({'OR': 'Genuine', 'CG': 'Fake'})
                    fig_fake = px.bar(auth_counts, x='label', y='count',
                                      color='label',
                                      color_discrete_map={'Genuine':'#6ee7b7','Fake':'#f97316'},
                                      text='count')
                    fig_fake.update_traces(textposition='outside')
                    fig_fake.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                            font_color='#e2e8f0', showlegend=False,
                                            xaxis=dict(color='#e2e8f0'), yaxis=dict(color='#e2e8f0'), margin=dict(t=20))
                    st.plotly_chart(fig_fake, use_container_width=True)

                # Rating Distribution
                r2c1, r2c2 = st.columns(2)

                with r2c1:
                    st.markdown('<div class="section-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
                    fig_rating = px.histogram(df_upload, x='rating', nbins=5,
                                              color_discrete_sequence=['#a78bfa'])
                    fig_rating.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                              font_color='#e2e8f0',
                                              xaxis=dict(color='#e2e8f0', title='Rating'),
                                              yaxis=dict(color='#e2e8f0', title='Count'), margin=dict(t=20))
                    st.plotly_chart(fig_rating, use_container_width=True)

                with r2c2:
                    st.markdown('<div class="section-title">🏷️ Fake Review % by Category</div>', unsafe_allow_html=True)
                    if len(df_upload['category_clean'].unique()) > 1:
                        cat_df = df_upload.groupby('category_clean').apply(
                            lambda x: (x['Predicted_Authenticity'] == 'CG').mean() * 100
                        ).reset_index(name='Fake %').sort_values('Fake %', ascending=True)
                        fig_cat = px.bar(cat_df, x='Fake %', y='category_clean', orientation='h',
                                         color='Fake %', color_continuous_scale='Purples')
                        fig_cat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                               font_color='#e2e8f0', coloraxis_showscale=False,
                                               yaxis=dict(color='#e2e8f0', title=''),
                                               xaxis=dict(color='#e2e8f0', title='Fake Review %'), margin=dict(t=20))
                        st.plotly_chart(fig_cat, use_container_width=True)
                    else:
                        st.info("Only one category detected in dataset.")

                # Word Clouds
                st.markdown('<div class="section-title">☁️ Word Clouds</div>', unsafe_allow_html=True)
                wc1, wc2 = st.columns(2)

                pos_text = " ".join(df_upload[df_upload['Predicted_Sentiment'] == 'Positive']['cleaned_review'].dropna())
                neg_text = " ".join(df_upload[df_upload['Predicted_Sentiment'] == 'Negative']['cleaned_review'].dropna())

                with wc1:
                    if pos_text.strip():
                        wc_pos = WordCloud(width=800, height=350, background_color='#064e3b',
                                           colormap='Greens', max_words=80).generate(pos_text)
                        fig, ax = plt.subplots(figsize=(8, 3.5))
                        fig.patch.set_facecolor('#064e3b')
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title("Positive Reviews", color='#6ee7b7', fontsize=14, pad=10)
                        st.pyplot(fig)
                    else:
                        st.info("No positive reviews to display.")

                with wc2:
                    if neg_text.strip():
                        wc_neg = WordCloud(width=800, height=350, background_color='#7f1d1d',
                                           colormap='Reds', max_words=80).generate(neg_text)
                        fig, ax = plt.subplots(figsize=(8, 3.5))
                        fig.patch.set_facecolor('#7f1d1d')
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title("Negative Reviews", color='#fca5a5', fontsize=14, pad=10)
                        st.pyplot(fig)
                    else:
                        st.info("No negative reviews to display.")

                st.markdown("---")
                st.markdown("### 📥 Download Results")
                csv_export = df_upload[['text_', 'rating', 'category', 'Predicted_Sentiment', 'Predicted_Authenticity']].to_csv(index=False)
                st.download_button(
                    label="📊 Download Analysis Results (CSV)",
                    data=csv_export,
                    file_name="analysis_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
