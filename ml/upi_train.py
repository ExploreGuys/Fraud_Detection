# ml/upi_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
print("Loading UPI dataset...")
try:
    df = pd.read_csv('ml/dummy_upi_transactions.csv')
except FileNotFoundError:
    print("Error: 'ml/dummy_upi_transactions.csv' not found. Make sure the file is in the 'ml' directory.")
    exit()

# Preprocessing
print("Preprocessing data...")
# Encode categorical features
label_encoders = {}
for column in ['payeeId', 'payerId', 'deviceId']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
features = ['amount', 'payeeId', 'payerId', 'deviceId']
target = 'isFraud'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
print("Training UPI fraud detection model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
print("Saving model to upi_model.pkl...")
with open('ml/upi_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("UPI model training complete and model saved.")
