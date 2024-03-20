import numpy as np
import joblib
import argparse

# Teminal agurments
parser = argparse.ArgumentParser(description='Predict maintainability, based on transition and previous maintainability rankings.')
parser.add_argument('transition', type=str, help='Transition as string, for example Attractive -> Terminal')
parser.add_argument('prev_analysability', type=int, help='Previous analysability ranking as integer')
parser.add_argument('prev_changeability', type=int, help='Previous  changeability ranking as integer')
parser.add_argument('prev_testability', type=int, help='Previous testability ranking as integer')
args = parser.parse_args()

# Load model and encoder
model = joblib.load('quality_prediction_linear_model_no_overall.joblib')
encoder = joblib.load('transition_encoder_no_overall.joblib')

# Use arguments
new_transition = args.transition
prev_values = [args.prev_analysability, args.prev_changeability, args.prev_testability]

# Transform transition with encoder
new_transition_encoded = encoder.transform([[new_transition]])

# Add prev maintainability rankings
new_data = np.hstack((new_transition_encoded, np.array(prev_values).reshape(1, -1)))

# Use model to predict
predicted_quality = model.predict(new_data)

# Show predicted maintainability
print("Predicted maintainability (analysability, changeability and testability):", predicted_quality)

