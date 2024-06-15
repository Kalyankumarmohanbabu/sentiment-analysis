import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import NotFittedError

# Function to load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('D:\\sentiment analysis\\sentiment_model.pkl')
        vectorizer = joblib.load('D:\\sentiment analysis\\vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

# Load the model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Data cleaning function
def clean_text(text):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Title of the app
st.title("Sentiment Analysis App")

# Text input
user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if model and vectorizer:
        try:
            # Clean the input text
            cleaned_text = clean_text(user_input)
            # Transform the text using the vectorizer
            vectorized_text = vectorizer.transform([cleaned_text])
            # Predict the sentiment
            prediction = model.predict(vectorized_text)
            # Map the prediction to sentiment label
            sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
            sentiment = sentiment_map[prediction[0]]

            # Display the sentiment
            st.write(f"Sentiment: **{sentiment}**")
        except NotFittedError as e:
            st.error(f"The model is not fitted yet: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Model or vectorizer not loaded. Please refresh the page.")
