import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load the IMDB movie review dataset
max_features = 10000
max_len = 200
(X_train, y_train), (_, _) = imdb.load_data(num_words=max_features)

# Preprocess the text input
def preprocess_text(text):
    word_index = imdb.get_word_index()
    words = text.lower().split()
    x = [word_index[word] if word in word_index and word_index[word] < max_features else 0 for word in words]
    x = sequence.pad_sequences([x], maxlen=max_len)
    return x

# Load the trained model
model = load_model('sentiment_model.h5')

# Sentiment prediction
def predict_sentiment(text):
    x = preprocess_text(text)
    prediction = model.predict(x)[0][0]
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"

# Streamlit app
def main():
    st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
    st.title("Sentiment Analysis")
    text = st.text_area("Enter a movie review", height=200)
    if st.button("Predict"):
        if text:
            sentiment = predict_sentiment(text)
            st.success(f"Sentiment: {sentiment}")

if __name__ == '__main__':
    main()
