import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence



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
    st.title("Sentiment Analysis")
    text = st.text_area("Enter a movie review", height=200)
    if st.button("Predict"):
        if text:
            sentiment = predict_sentiment(text)
            st.success(f"Sentiment: {sentiment}")

if __name__ == '__main__':
    main()
