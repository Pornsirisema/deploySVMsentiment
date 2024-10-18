
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import streamlit as st
import pickle
import pandas as pd

# Load the SVM model and vectorizer
with open('model_SVM_deploy_sentiment.pkl', 'rb') as file:
    svm_model, vectorizer = pickle.load(file)

# Streamlit interface for input
st.title("Sentiment Analysis using SVM")
user_input = st.text_input("Enter your comment:", "ได้รับของไวมาก สินค้าตรงปกค่ะ")

if user_input:
    # Prepare the input data
    x_new = pd.DataFrame({'ข้อความ': [user_input]})
    X_vectorized = vectorizer.transform(x_new['ข้อความ'])
    
    # Predict sentiment
    y_pred = svm_model.predict(X_vectorized)
    
    # Display the result
    st.write(f"Predicted sentiment: {y_pred[0]}")
