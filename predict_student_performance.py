# predict_student_pass.py

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('random_forest_student_pass_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

# Sample new data (ensure it follows the same structure)
# Corrected 'test_preparation_course' to use 'none' or 'completed'
new_data = pd.DataFrame({
    'gender': ['male'],
    'race_ethnicity': ['Group B'],
    'parental_level_of_education': ['bachelor\'s degree'],
    'lunch': ['standard'],
    'test_preparation_course': ['none'],  # Use 'none' or 'completed'
    'math_score': [80],
    'reading_score': [80],
    'writing_score': [80],
})

# Handle categorical variables
new_data = pd.get_dummies(new_data, columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'], drop_first=True)

# Ensure all expected columns are present
# Missing columns will be filled with 0
expected_columns = model.feature_names_in_
for col in expected_columns:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[expected_columns]

# Feature scaling using the loaded scaler
new_data[['math_score', 'reading_score', 'writing_score']] = scaler.transform(new_data[['math_score', 'reading_score', 'writing_score']])

# Make prediction
prediction = model.predict(new_data)
prediction_proba = model.predict_proba(new_data)[:,1]

print(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Probability of Passing: {prediction_proba[0]:.2f}")
