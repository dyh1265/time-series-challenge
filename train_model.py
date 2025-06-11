import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from utils.utils import setSeed
from prepare_data import prepare_accident_data

# Set seed for reproducibility
setSeed(42)

# Prepare the data
_, y_real, data = prepare_accident_data('data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv')

# Filter the data for alcohol-related accidents
alkoholunfaelle_data = data[data['MONATSZAHL'] == 'Alkoholunfälle']
alkoholunfaelle_data = alkoholunfaelle_data[alkoholunfaelle_data['MONAT'] != 'Summe']
month_data = alkoholunfaelle_data['MONAT']
month_data = month_data.str[-2:]
alkoholunfaelle_data['MONAT'] = month_data
alkoholunfaelle_data['MONAT'] = pd.to_numeric(alkoholunfaelle_data['MONAT'], errors='coerce')
alkoholunfaelle_data = alkoholunfaelle_data[alkoholunfaelle_data['AUSPRAEGUNG'] == 'insgesamt']
alkoholunfaelle_data = alkoholunfaelle_data[['JAHR', 'MONAT', 'WERT']]

# Features and target
X = alkoholunfaelle_data[['JAHR', 'MONAT']].values
y = alkoholunfaelle_data['WERT'].values

# Normalize the features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train.ravel())

# Predict on the test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_inverse = scaler_y.inverse_transform(y_test)

# Compute RMSE
rmse = root_mean_squared_error(y_test_inverse, y_pred)
print(f"RMSE for the alcohol-related accidents prediction model: {rmse}")

# Plot actual vs predicted values for the test set
plt.scatter(y_test_inverse, y_pred, label='Predictions')
plt.plot([y_test_inverse.min(), y_test_inverse.max()], [y_test_inverse.min(), y_test_inverse.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()

# Predict for January 2021
next_month = np.array([[2021, 1]])
next_month_scaled = scaler_X.transform(next_month)
pred_next_month_scaled = model.predict(next_month_scaled)
pred_next_month = scaler_y.inverse_transform(pred_next_month_scaled.reshape(-1, 1))
print(f"Predicted number of alcohol-related accidents for January 2021: {pred_next_month[0][0]}")
print(f"Real number of alcohol-related accidents for January 2021: {y_real}")
rmse = root_mean_squared_error([y_real], pred_next_month)
print(f"RMSE for the prediction of January 2021: {rmse}")

# Save the model and scalers
with open('models/linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("Linear regression model and scalers saved.")