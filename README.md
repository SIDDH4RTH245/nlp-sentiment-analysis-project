# 🧠 NLP Sentiment Analysis Project

## 📌 Overview
This project builds a Natural Language Processing (NLP) model to classify movie reviews as **positive** or **negative** using machine learning techniques.

The goal is to understand and analyze text data to extract sentiment and build an AI-powered classification system.

---

## 🎯 Objective
- Perform text preprocessing on raw reviews
- Convert text into numerical features using TF-IDF
- Train machine learning models for classification
- Evaluate model performance using appropriate metrics
- Deploy the model using a Streamlit web application

---

## 📊 Dataset
- Source: IMDB Movie Reviews Dataset (50K reviews)
- Each review is labeled as:
  - **Positive (1)**
  - **Negative (0)**

---

## ⚙️ Project Workflow

### 1️⃣ Data Cleaning
- Removed missing values
- Converted sentiment labels into numerical format

### 2️⃣ Exploratory Data Analysis (EDA)
- Sentiment distribution
- Review length analysis

### 3️⃣ Text Preprocessing
- Lowercasing
- Removing stopwords
- Removing punctuation
- Tokenization

### 4️⃣ Feature Engineering
- TF-IDF Vectorization
- Converted text into numerical vectors

### 5️⃣ Model Training
- Logistic Regression
- Trained on processed text data

### 6️⃣ Model Evaluation
- Accuracy
- Precision & Recall
- Confusion Matrix

### 7️⃣ Model Deployment
- Built a Streamlit app for real-time sentiment prediction

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- NLP (TF-IDF)
- Streamlit
