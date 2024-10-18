import streamlit as st
import pickle
import pandas as pd

# Load the SVM model and vectorizer
with open('model_SVM_deploy_sentiment.pkl', 'rb') as file:
    svm_model, vectorizer = pickle.load(file)

# Streamlit interface for input
st.title("Sentiment Analysis using SVM")

# Option to input text manually
st.subheader("Input a single comment for sentiment prediction")
user_input = st.text_input("Enter your comment:", "")

# Option to upload a CSV file
st.subheader("Or upload a CSV file for bulk prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to predict sentiment
def predict_sentiment(texts):
    x_new = pd.DataFrame({'ข้อความ': texts})
    X_vectorized = vectorizer.transform(x_new['ข้อความ'])
    predictions = svm_model.predict(X_vectorized)
    x_new['Predicted Sentiment'] = predictions
    return x_new

# If user provides text input
if user_input:
    result = predict_sentiment([user_input])
    st.write("Prediction for the input comment:")
    st.write(result[['ข้อความ', 'Predicted Sentiment']])

# If user uploads a CSV file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'ข้อความ' in df.columns:
        st.write("Uploaded file preview:")
        st.write(df.head())
        
        # Perform sentiment prediction
        result_df = predict_sentiment(df['ข้อความ'].tolist())
        
        # Display result
        st.write("Prediction results:")
        st.write(result_df)

        # Button to download the result as CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download prediction results as CSV",
            data=csv,
            file_name='sentiment_predictions.csv',
            mime='text/csv'
        )
    else:
        st.write("The uploaded file must contain a 'ข้อความ' column.")
