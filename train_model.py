
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "url_features_extracted1.csv"
MODEL_PATH = BASE_DIR / "phishing_model.joblib"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# Remove rows without labels
df = df.dropna(subset=["ClassLabel"]).copy()

# Remove duplicate rows
df = df.drop_duplicates()

# Normalize column names
df.columns = df.columns.str.strip()

# URL and label columns
URL_COLUMN = "URL"
LABEL_COLUMN = "ClassLabel"

# Remove rows without URLs
df = df.dropna(subset=[URL_COLUMN]).copy()

# Convert labels to integers
df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN], errors="coerce"
)
df = df.dropna(subset=[LABEL_COLUMN]).copy()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# Use only numeric feature columns.
# Never use the raw URL or target label as a feature.
excluded_columns = {URL_COLUMN, LABEL_COLUMN}

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[column])
]

if not feature_columns:
    raise ValueError(
        "No numeric feature columns found. "
        "Check the CSV structure."
    )

# Replace infinite values and missing feature values
X = df[feature_columns].copy()
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median()).fillna(0)

y = df[LABEL_COLUMN]

print("\nLabel distribution:")
print(y.value_counts().sort_index())

print("\nFeatures used:")
for column in feature_columns:
    print("-", column)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Evaluate on held-out data
predictions = model.predict(X_test)

print("\n========== MODEL RESULTS ==========")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and feature metadata together
artifact = {
    "model": model,
    "feature_columns": feature_columns,
    "label_mapping": {
        0: "Phishing",
        1: "Legitimate",
    },
}

joblib.dump(artifact, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training completed.")
