import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
print("Loading training data...")
train_data = pd.read_csv("data/cs-training.csv")
print("Training data loaded. Shape:", train_data.shape)

# Preprocess data
from a_preprocessing_featurepipeline import preprocess_data
print("Starting preprocessing of training data...")
train_data = preprocess_data(train_data)
print("Preprocessing completed. Processed data shape:", train_data.shape)

# Ensure the target column is correctly named and exists
if 'SeriousDlqin2yrs' not in train_data.columns:
    raise KeyError("The target column 'SeriousDlqin2yrs' is missing in the training data.")

# Separate features and target
print("Separating features and target...")
X_train = train_data.drop(columns=["SeriousDlqin2yrs"])
y_train = train_data["SeriousDlqin2yrs"]
print("Features shape:", X_train.shape, "Target shape:", y_train.shape)

# Train model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Save model
print("Saving the model...")
joblib.dump(model, "models/random_forest_model.pkl")
print("Model saved successfully.")
