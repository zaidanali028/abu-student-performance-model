# student_performance_model.py

import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Initialize Faker for realistic data
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Step 1: Generate Synthetic Dataset ---

# Define the number of samples
num_samples = 1000

# Define possible categories
genders = ['male', 'female']
race_ethnicities = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
parental_education_levels = [
    'high school', 'some college', "associate's degree",
    "bachelor's degree", "master's degree"
]
lunch_options = ['standard', 'free/reduced']
test_prep_courses = ['none', 'completed']

# Function to generate scores with some randomness
def generate_score(mean=65, std=15):
    score = np.random.normal(mean, std)
    return int(min(max(score, 0), 100))  # Ensure score is between 0 and 100

# Generate data
data = {
    'gender': [random.choice(genders) for _ in range(num_samples)],
    'race_ethnicity': [random.choice(race_ethnicities) for _ in range(num_samples)],
    'parental_level_of_education': [random.choice(parental_education_levels) for _ in range(num_samples)],
    'lunch': [random.choice(lunch_options) for _ in range(num_samples)],
    'test_preparation_course': [random.choice(test_prep_courses) for _ in range(num_samples)],
    'math_score': [generate_score(mean=65, std=15) for _ in range(num_samples)],
    'reading_score': [generate_score(mean=70, std=10) for _ in range(num_samples)],
    'writing_score': [generate_score(mean=68, std=12) for _ in range(num_samples)],
}

df = pd.DataFrame(data)

# Define 'Pass' based on math_score
df['Pass'] = df['math_score'].apply(lambda x: 1 if x >= 50 else 0)

# Optional: Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display first few rows
print("First five rows of the synthetic dataset:")
print(df.head())

# Save to CSV
csv_filename = 'synthetic_student_performance.csv'
df.to_csv(csv_filename, index=False)
print(f"\nSynthetic dataset '{csv_filename}' has been created successfully!")

# --- Step 2: Load the Dataset ---
# (In this case, we have just created and saved it. To load it, we can read from the saved CSV.)
df = pd.read_csv(csv_filename)

# --- Step 3: Define Target Variable ---
# (Already defined as 'Pass' in the dataset)

# --- Step 4: Handle Categorical Variables ---
categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Step 5: Feature Scaling ---
scaler = StandardScaler()
score_cols = ['math_score', 'reading_score', 'writing_score']
df[score_cols] = scaler.fit_transform(df[['math_score', 'reading_score', 'writing_score']])

# Save the scaler for future use
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as '{scaler_filename}'.")

# --- Step 6: Split the Dataset ---
X = df.drop(['Pass'], axis=1)
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# --- Step 7: Train a Model ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# --- Step 8: Make Predictions ---
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

# --- Step 9: Evaluate the Model ---
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

# --- Step 10: Feature Importance ---
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances - Random Forest')
plt.tight_layout()
plt.show()

# --- Step 11: Save the Model ---
model_filename = 'random_forest_student_pass_model.pkl'
joblib.dump(rf, model_filename)
print(f"\nTrained Random Forest model saved as '{model_filename}'.")
