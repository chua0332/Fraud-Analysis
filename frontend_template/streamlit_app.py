import uvicorn
import numpy as np
import pandas as pd
import pickle
import streamlit as st


#Load both the model and the preprocessor pkl files
model_in = open('/Users/eugenechua/Downloads/skillsfuture_interview/fraud/Fraud-Analysis/XGB_classifier.pkl','rb')
XGBmodel = pickle.load(model_in)
preprocessor_in = open('/Users/eugenechua/Downloads/skillsfuture_interview/fraud/Fraud-Analysis/preprocessor.pkl','rb')
preprocessor = pickle.load(preprocessor_in)


def main():
    st.title("Fraud Prediction App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h3 style="color:white;text-align:center;">This app is powered by XGboost :) </h3>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    category = st.text_input("category","Type Here")
    amt = st.text_input("amount","Type Here")
    gender = st.text_input("gender","Type Here")
    job = st.text_input("job","Type Here")
    merchant = st.text_input("merchant","Type Here")
    age = st.text_input("age","Type Here")
    input_df = {'category': category,
                'amt': amt,
                'gender': gender,
                'job': job,
                'merchant': merchant,
                'age': age}
    input_df = pd.DataFrame(input_df, index=[0])
    X_processed = preprocessor.transform(input_df)
    if st.button("Predict"):
        y_pred=XGBmodel.predict(X_processed)
        if y_pred == 0:
            st.success('There is no fraud')
        if y_pred == 1:
            st.warning('There is fraud!')

if __name__=='__main__':
    main()
    