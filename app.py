import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('sentiment_model.sav', 'rb') as model_file:
    classifier_nb = pickle.load(model_file)

# Function to predict sentiment
def predict_sentiment(text, model):
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    probabilities = model.predict_proba([text])[0]

    # Determine sentiment based on highest probability
    max_prob_index = np.argmax(probabilities)  # Index of the max probability
    prediction = [-1, 0, 1][max_prob_index]  # Map index to sentiment
    sentiment = sentiment_map.get(prediction, 'Unknown')

    return sentiment, probabilities

# Streamlit UI
st.title("Sentiment Prediction App")
st.write("Analyze the sentiment of your text (Positive, Neutral, or Negative).")

# Input text box
user_input = st.text_area("Enter text:", "")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment, probabilities = predict_sentiment(user_input, classifier_nb)
        st.subheader(f"Predicted Sentiment: {sentiment}")
        st.write(f"Probabilities:")
        st.write(f"- Negative: {probabilities[0]:.2f}")
        st.write(f"- Neutral: {probabilities[1]:.2f}")
        st.write(f"- Positive: {probabilities[2]:.2f}")
    else:
        st.error("Please enter some text to analyze.")
