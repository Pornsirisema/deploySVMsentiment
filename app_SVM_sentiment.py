import streamlit as st
import pickle
import pandas as pd

# Load the SVM model and vectorizer
with open('model_SVM_deploy_sentiment.pkl', 'rb') as file:
    svm_model, vectorizer = pickle.load(file)

# Streamlit interface for input
st.title("Sentiment Analysis using SVM")

# User can either input text or upload an Excel file
option = st.radio("Choose input method:", ('Text Input', 'Upload Excel File'))

if option == 'Text Input':
    # Text input
    user_input = st.text_input("Enter your comment:", "ได้รับของไวมาก สินค้าตรงปกค่ะ")
    
    if user_input:
        # Prepare the input data
        x_new = pd.DataFrame({'ข้อความ': [user_input]})
        X_vectorized = vectorizer.transform(x_new['ข้อความ'])
        
        # Predict sentiment
        y_pred = svm_model.predict(X_vectorized)
        
        # Display the result
        st.write(f"Predicted sentiment: {y_pred[0]}")

elif option == 'Upload Excel File':
    # File upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    print(uploaded_file)
    
    if uploaded_file:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        if 'ข้อความ' in df.columns:
            # Vectorize the text data
            X_vectorized = vectorizer.transform(df['ข้อความ'])
            
            # Predict sentiment
            df['Predicted Sentiment'] = svm_model.predict(X_vectorized)
            
            # Show the predictions
            st.write(df)
            
            # Allow user to download the prediction results
            @st.cache_data
            # def convert_df(df):
            #     return df.to_excel(index=False)
            
            # # Convert dataframe to Excel format for download
            # result = convert_df(df)

            def convert_df_to_excel(df):
                # Create a buffer to hold the excel data
                output = io.BytesIO()
                # Write the DataFrame to the buffer
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                # Get the Excel file data
                output.seek(0)
                return output
            
            # Convert dataframe to Excel format for download
            result = convert_df_to_excel(df)
            
            # Download button
            st.download_button(
                label="Download predictions as Excel",
                data=result,
                file_name="sentiment_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("The uploaded file must contain a 'ข้อความ' column.")
