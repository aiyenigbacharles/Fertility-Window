# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the model columns
#odel = joblib.load('fertility_model.pkl')
model = joblib.load('model_fertility.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Fertility Window Predictor", page_icon="👶")

# Page Title
st.title('👶 Fertility Window Predictor')
st.write("Enter your daily physiological data to predict your fertility status. This tool helps in family planning by identifying the most fertile days in your menstrual cycle.")


# User input section in the sidebar
st.sidebar.header('Enter Your Data:')

cycle_day = st.sidebar.number_input('Day of Cycle', min_value=1, max_value=60, value=14, step=1)
bbt = st.sidebar.number_input('Basal Body Temperature (°C)', min_value=35.0, max_value=38.0, value=36.5, step=0.01, format="%.2f")
cervical_mucus = st.sidebar.selectbox('Cervical Mucus Type', ['Dry', 'Sticky', 'Creamy', 'Watery', 'Egg White'])

# Prediction function
def predict(cycle_day, bbt, cervical_mucus):
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'cycle_day': [cycle_day],
        'bbt': [bbt],
        'cervical_mucus': [cervical_mucus]
    })
    
    # One-hot encode the categorical feature
    input_encoded = pd.get_dummies(input_data, columns=['cervical_mucus'])
    
    # Align the columns with the training columns
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(input_aligned)
    prediction_proba = model.predict_proba(input_aligned)
    
    return prediction[0], prediction_proba[0]

# Predict and display the result
if st.sidebar.button('Predict Fertility Status'):
    prediction, prediction_proba = predict(cycle_day, bbt, cervical_mucus)
    
    st.subheader('Prediction Result')
    if prediction == 1:
        probability = prediction_proba[1] * 100
        st.success(f'✅ You are likely in your **Fertile Window**.')
        st.write(f"Confidence: **{probability:.2f}%**")
        st.progress(int(probability))
        st.info("This is a high-fertility day. The chances of conception are highest during this period.")
    else:
        probability = prediction_proba[0] * 100
        st.error(f'❌ You are likely **Not** in your Fertile Window.')
        st.write(f"Confidence: **{probability:.2f}%**")
        st.progress(int(probability))
        st.info("This is a low-fertility day. Conception is less likely.")

st.write("---")
st.write("**Disclaimer:** This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice.")