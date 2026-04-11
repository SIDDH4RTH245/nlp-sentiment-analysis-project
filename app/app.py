import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "sentiment_model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "..", "model", "vectorizer.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        st.warning("Models not found. Training model now (this may take a minute)...")
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "imdb_cleaned.csv")
        df = pd.read_csv(data_path).dropna()
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(df['review'])
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tfidf, df['sentiment'])
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
    return joblib.load(model_path), joblib.load(vectorizer_path)

model, vectorizer = load_models()

st.title("🎬 Sentiment Analysis App")

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    transformed = vectorizer.transform([review])
    prediction = model.predict(transformed)

    if prediction[0] == 1:
        st.success("😊 Positive Review")
    else:
        st.error("😡 Negative Review")