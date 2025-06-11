import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import random
import pickle
from utils.utils import setSeed
import keras
# Load the prepared data
from prepare_data import prepare_accident_data


def create_model():
    setSeed(42)  # Set a seed for reproducibility
    model = Sequential([
        Dense(10, activation='relu', kernel_regularizer='l2'),
        Dense(1, activation='linear')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare the data
_, y_real, data = prepare_accident_data('data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv')

# Filter the data for the first month of each year
alkoholunfaelle_data = data[data['MONATSZAHL'] == 'Alkoholunfälle']
# drop the Monat having the value 'Summe'
alkoholunfaelle_data = alkoholunfaelle_data[alkoholunfaelle_data['MONAT'] != 'Summe']
month_data = alkoholunfaelle_data['MONAT']
# only keep the laset two digits of the month
month_data = month_data.str[-2:]
alkoholunfaelle_data['MONAT'] = month_data
# Convert the 'MONAT' column to numeric values
alkoholunfaelle_data['MONAT'] = pd.to_numeric(alkoholunfaelle_data['MONAT'], errors='coerce')

alkoholunfaelle_data = alkoholunfaelle_data[alkoholunfaelle_data['AUSPRAEGUNG'] == 'insgesamt']

# Only keep the relevant columns
alkoholunfaelle_data = alkoholunfaelle_data[['JAHR', 'MONAT', 'WERT']]

# keep only JAHR and MONAT as features
X = alkoholunfaelle_data[['JAHR', 'MONAT']].values
# keep only NEXT_WERT as target
y = alkoholunfaelle_data['WERT'].values

# Normalize the features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
# Normalize the target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# create a train and test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
# Build the model
setSeed(42)  # Set a seed for reproducibility
model = create_model()
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=64, 
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)
# Predict for the test set
y_pred_scaled = model.predict(X_test)
# Inverse transform the predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)
# Inverse transform the test set
y_test_inverse = scaler_y.inverse_transform(y_test)
# Compute RMSE
rmse = root_mean_squared_error(y_test_inverse, y_pred)
print(f"RMSE for the alcohol-related accidents prediction model: {rmse}")
# plot the training history
plt.plot(history.history['loss'], label='Training Loss')    
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# make predictions for the next month
next_month = np.array([[2021, 1]])  # January 2021
next_month_scaled = scaler_X.transform(next_month)
pred_next_month_scaled = model.predict(next_month_scaled)
# Inverse transform the prediction
pred_next_month = scaler_y.inverse_transform(pred_next_month_scaled)
print(f"Predicted number of alcohol-related accidents for January 2021: {pred_next_month[0][0]}")
print(f"Real number of alcohol-related accidents for January 2021: {y_real}")
# compute the RMSE
rmse = root_mean_squared_error([y_real], pred_next_month)
print(f"RMSE for the prediction of January 2021: {rmse}")

# Save the scalers
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
# Save the training history
model.save('models/model.keras')


# Load model
model = tf.keras.models.load_model('models/model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Optimized TFLite model saved to model.tflite")


# Load the TensorFlow model
tf_model = tf.keras.models.load_model(
    'models/model.keras',
    custom_objects={'mse': tf.keras.metrics.MeanSquaredError()}
)
pred_next_month_scaled = tf_model.predict(next_month_scaled)
# Inverse transform the prediction
pred_next_month = scaler_y.inverse_transform(pred_next_month_scaled)
print(f"Predicted number of alcohol-related accidents for January 2021: {pred_next_month[0][0]}")
print(f"Real number of alcohol-related accidents for January 2021: {y_real}")