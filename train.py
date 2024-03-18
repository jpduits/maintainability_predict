import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib

file_path = 'dump.csv'
data = pd.read_csv(file_path)

# Filter rows
required_columns = [
    'Transition', 
    'prev_sig_analysability_ranking_no_unit_tests',
    'prev_sig_changeability_ranking_no_unit_tests',
    'prev_sig_testability_ranking_no_unit_tests'
]
data = data.dropna(subset=required_columns)

# one-hot encoding for 'Transition' 
encoder = OneHotEncoder(sparse=False)
transitions_encoded = encoder.fit_transform(data[['Transition']])

# Combine Transition with 'prev_' columns
prev_columns = data[[
    'prev_sig_analysability_ranking_no_unit_tests', 
    'prev_sig_changeability_ranking_no_unit_tests', 
    'prev_sig_testability_ranking_no_unit_tests'
]].values
X = np.hstack((transitions_encoded, prev_columns))

# Target columns
y = data[[
    'sig_analysability_ranking_no_unit_tests', 
    'sig_changeability_ranking_no_unit_tests', 
    'sig_testability_ranking_no_unit_tests'
]]

# Splits the data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict quality for test set
y_pred = model.predict(X_test)

# RMSE for every target
# Root Mean Squared Error
rmse = {y.columns[i]: np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])) for i in range(y_test.shape[1])}

# Print de RMSE waarden
print("RMSE analysability, changeability and testability:", rmse)

# Save model and encoder
joblib.dump(model, 'quality_prediction_linear_model_no_overall.joblib')
joblib.dump(encoder, 'transition_encoder_no_overall.joblib')

