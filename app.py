import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model and scaler
model = tf.keras.models.load_model('aqi_model.h5')
scaler = joblib.load('scaler.pkl')

st.title('Air Quality Index Prediction')

# Input fields
PM10 = st.number_input('PM10', min_value=0.0)
PM25 = st.number_input('PM2.5', min_value=0.0)
NO2 = st.number_input('NO2', min_value=0.0)
O3 = st.number_input('O3', min_value=0.0)
CO = st.number_input('CO', min_value=0.0)
SO2 = st.number_input('SO2', min_value=0.0)
NH3 = st.number_input('NH3', min_value=0.0)

# Prediction function
def predict_aqi(features):
    try:
        # Convert input features to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the input features using the loaded scaler
        features_scaled = scaler.transform(features_array)
        
        # Make prediction using the loaded model
        prediction = model.predict(features_scaled)
        predicted_aqi = prediction[0][0]
        
        return float(predicted_aqi)
    except Exception as e:
        return str(e)

# Predict button
if st.button('Predict AQI'):
    features = [PM10, PM25, NO2, O3, CO, SO2, NH3]
    result = predict_aqi(features)
    st.success(f'Predicted AQI: {result}')
