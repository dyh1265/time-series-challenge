from flask import Flask, request, jsonify
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the TensorFlow model
tf_model = tf.keras.models.load_model(
    'models/alcohol_accidents_prediction_model.h5',
    custom_objects={'mse': tf.keras.metrics.MeanSquaredError()}
)

# Load scalers
with open('models/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('models/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate input
        year = data.get('year')
        month = data.get('month')
        if year is None or month is None:
            return jsonify({'error': 'Missing year or month in request'}), 400
        try:
            year = float(year)
            month = float(month)
        except (ValueError, TypeError):
            return jsonify({'error': 'Year and month must be numeric'}), 400

        # Scale the input data
        input_data = np.array([[year, month]])
        input_data_scaled = scaler_X.transform(input_data)

        # Predict using the loaded model
        prediction_scaled = tf_model.predict(input_data_scaled)

        # Inverse transform the prediction
        prediction = scaler_y.inverse_transform(prediction_scaled)

        # Return prediction as a list for JSON serialization
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use PORT environment variable for Render compatibility
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)