import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset/cleaned_imdb.csv")

# Features and labels
X = df['review']
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(X_train)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Streamlit UI
st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter a movie review:")

if st.button("Predict"):

    review_vector = tfidf.transform([review])

    prediction = model.predict(review_vector)

    st.success(f"Sentiment: {prediction[0]}")