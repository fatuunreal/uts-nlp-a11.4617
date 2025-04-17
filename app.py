import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to load the model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to load evaluation data
def load_evaluation_data(data_path):
    try:
        data = joblib.load(data_path)
        return data
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return None

# Function to predict sentiment
def predict_sentiment(text, model):
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    probabilities = model.predict_proba([text])[0]

    # Determine sentiment based on highest probability
    max_prob_index = np.argmax(probabilities)  # Index of the max probability
    prediction = [-1, 0, 1][max_prob_index]  # Map index to sentiment
    sentiment = sentiment_map.get(prediction, 'Unknown')

    return sentiment, probabilities, prediction

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Streamlit UI
st.title("Sentiment Analysis of Tweets Fufufafa Using Naive Bayes Algorithm")
st.write("Input text (Positive, Neutral, or Negative).")

# Load the model
model_path = 'sentiment_model.sav'  # Path to the saved model
model = load_model(model_path)

# Load evaluation data
evaluation_data_path = 'evaluation_data.sav'  # Path to the saved evaluation data
evaluation_data = load_evaluation_data(evaluation_data_path)

# Static accuracy metrics
accuracy = 0.6809
recall = 0.6809
precision = 0.6695
f1 = 0.6731

if model is not None and evaluation_data is not None:
    # Input text box
    user_input = st.text_area("Enter text:", "")

    # Predict button
    if st.button("Predict Sentiment"):
        if user_input.strip() != "":
            # Predict sentiment
            sentiment, probabilities, prediction = predict_sentiment(user_input, model)
            
            # Display predicted sentiment
            st.subheader(f"Predicted Sentiment: {sentiment}")
            
            # Display probabilities
            st.write("**Probabilities:**")
            st.write(f"- Negative: {probabilities[0]:.2f}")
            st.write(f"- Neutral: {probabilities[1]:.2f}")
            st.write(f"- Positive: {probabilities[2]:.2f}")

            # Visualisasi distribusi probabilitas (Pie Chart)
            st.subheader("Probability Distribution (Pie Chart)")
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Probability': probabilities
            })
            
            # Plot pie chart
            fig, ax = plt.subplots()
            ax.pie(prob_df['Probability'], labels=prob_df['Sentiment'], autopct='%1.1f%%', colors=['red', 'blue', 'green'])
            ax.set_title('Sentiment Probability Distribution')
            st.pyplot(fig)

            # Display static accuracy metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"- **Accuracy Test set:** {accuracy:.4f}")
            st.write(f"- **Recall Test set:** {recall:.4f}")
            st.write(f"- **Precision Test set:** {precision:.4f}")
            st.write(f"- **F1 Test set:** {f1:.4f}")
    
    # Display ROC Curve from evaluation data
    st.subheader("ROC Curve (Evaluation Data)")
    y_true = evaluation_data['y_true']
    y_prob = evaluation_data['y_prob']
    plot_roc_curve(y_true, y_prob)

else:
    st.error("Model or evaluation data could not be loaded. Please check the files.")
