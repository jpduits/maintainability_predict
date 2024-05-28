import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib

file_path = 'dump.csv'  
data = pd.read_csv(file_path)

# required columns
required_columns = [
    'oss_category', 
    'prev_oss_category',
    'prev_sig_analysability_ranking_no_unit_tests',
    'prev_sig_changeability_ranking_no_unit_tests',
    'prev_sig_testability_ranking_no_unit_tests',
    'sig_analysability_ranking_no_unit_tests', 
    'sig_changeability_ranking_no_unit_tests', 
    'sig_testability_ranking_no_unit_tests'
]
data = data.dropna(subset=required_columns)

# OneHotEncoder oss_categories
categories = ['Terminal', 'Stagnant', 'Fluctuating', 'Attractive']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore') # skip if unknown
encoder.fit(data[['oss_category', 'prev_oss_category']])

oss_categories_encoded = encoder.transform(data[['oss_category', 'prev_oss_category']])

prev_columns = data[[
    'prev_sig_analysability_ranking_no_unit_tests', 
    'prev_sig_changeability_ranking_no_unit_tests', 
    'prev_sig_testability_ranking_no_unit_tests'
]].values
X = np.hstack((oss_categories_encoded, prev_columns)) # np merges data to single array X

# Target columns (y)
y = data[[
    'sig_analysability_ranking_no_unit_tests', 
    'sig_changeability_ranking_no_unit_tests', 
    'sig_testability_ranking_no_unit_tests'
]]

# Split in dataset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train) # Train model

# Predict with testset
y_pred = model.predict(X_test)

# RMSE - Root Mean Squared Error
rmse = {y.columns[i]: np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])) for i in range(y_test.shape[1])}

# Print RMSE
print("RMSE for analysability, changeability en testability:", rmse)


# R-kwadraat
r_squared = r2_score(y_test, y_pred)
print(f"R-kwadraat voor het model: {r_squared}")


joblib.dump(model, 'quality_prediction_linear_model_with_oss_categories.joblib')
joblib.dump(encoder, 'oss_category_encoder_with_oss_categories.joblib')

