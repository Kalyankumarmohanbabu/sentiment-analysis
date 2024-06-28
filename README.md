Sentiment Analysis Web App

This project implements a sentiment analysis web application using Streamlit, scikit-learn, and joblib. The application allows users to enter text and predicts the sentiment of the text as either Positive, Negative, or Neutral.

Features:

* Input text area for user input.
* Sentiment analysis prediction using a pre-trained Multinomial Naive Bayes model.
* Text preprocessing for data cleaning (removing URLs, mentions, hashtags, special characters, and numbers).
* Integration of TF-IDF vectorization for text transformation.
* Interactive web interface powered by Streamlit for seamless user interaction.
Technologies Used:

* Streamlit framework for building interactive web applications in Python.
* scikit-learn library for machine learning model (Multinomial Naive Bayes) implementation.
* joblib for model and vectorizer serialization.
* Python for application logic and server-side processing.
Setup Instructions:

1 Clone the repository to your local machine.
2 Install the required dependencies using pip install -r requirements.txt.
3 Run the Streamlit application using streamlit run app.py.
4 Access the web application at the provided URL (usually http://localhost:8501) in your browser.
File Structure:

* app.py: Main Streamlit application code.
* requirements.txt: List of Python dependencies.
* sentiment_model.pkl: Pre-trained Multinomial Naive Bayes model for sentiment analysis.
* vectorizer.pkl: TF-IDF Vectorizer for text transformation.
Contributing:
Contributions to the project are welcome! You can contribute by submitting bug fixes, feature enhancements, or documentation improvements through pull requests.

