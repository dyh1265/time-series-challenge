from flask import Flask, request, jsonify
import pickle
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load the model tensorlfow model from a file
tf_model = tf.keras.models.load_model('models/alcohol_accidents_prediction_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        year = data['year']
        month = data['month']
        # load the scaler from a file
        with open('models/scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('models/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        # Scale the input data
        input_data = np.array([[year, month]]) 
        input_data_scaled = scaler_X.transform(input_data)
        # Predict using the loaded model
        model = tf_model
        prediction_scaled = model.predict(input_data_scaled)
        # Inverse transform the prediction
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        # Return prediction in required format
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)