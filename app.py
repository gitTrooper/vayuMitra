from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the model and scaler
model = tf.keras.models.load_model("aqi_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict_aqi():
    data = request.json
    features = np.array([data['PM10'], data['PM2.5'], data['NO2'], data['O3'], data['CO'], data['SO2'], data['NH3']])
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return jsonify({'AQI': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
