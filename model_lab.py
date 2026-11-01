
from pathlib import Path
import time

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "url_features_extracted1.csv"
RESULTS_PATH = BASE_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="PhishGuard AI | Model Lab",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Model Lab")
st.write(
    "Compare machine-learning algorithms for phishing URL classification."
)

st.info(
    "This experiment uses a random holdout split. Similar or related URLs "
    "may appear in both training and testing sets, so the results do not "
    "necessarily represent performance on unseen domains."
)

@st.cache_data
def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["URL", "ClassLabel"])
    df = df.drop_duplicates()

    df["ClassLabel"] = pd.to_numeric(
        df["ClassLabel"], errors="coerce"
    )
    df = df.dropna(subset=["ClassLabel"])
    df["ClassLabel"] = df["ClassLabel"].astype(int)

    feature_columns = [
        column for column in df.columns
        if column not in ["URL", "ClassLabel"]
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    X = df[feature_columns].copy()
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    y = df["ClassLabel"]

    return X, y, feature_columns


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

X, y, feature_columns = prepare_data()

st.metric("Dataset records", f"{len(X):,}")
st.metric("Numeric features", len(feature_columns))

if st.button("🚀 Run Model Comparison", type="primary"):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
            ),
        ),
        "Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(random_state=42),
        ),
    }

    results = []

    progress = st.progress(0)
    status = st.empty()

    for index, (name, model) in enumerate(models.items(), start=1):
        status.write(f"Training {name}...")

        start = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        elapsed = time.time() - start

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (phishing)": precision_score(
                y_test, predictions, pos_label=0, zero_division=0
            ),
            "Recall (phishing)": recall_score(
                y_test, predictions, pos_label=0, zero_division=0
            ),
            "F1-score (phishing)": f1_score(
                y_test, predictions, pos_label=0, zero_division=0
            ),
            "Training + prediction time (seconds)": round(elapsed, 2),
        })

        progress.progress(index / len(models))

    status.success("All four models evaluated!")

    results_df = pd.DataFrame(results).sort_values(
        "F1-score (phishing)", ascending=False
    )

    results_df.to_csv(RESULTS_PATH, index=False)

    st.subheader("📊 Model Performance")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision (phishing)": "{:.4f}",
            "Recall (phishing)": "{:.4f}",
            "F1-score (phishing)": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Accuracy Comparison")
    chart_data = results_df.set_index("Model")[["Accuracy"]]
    st.bar_chart(chart_data, y="Accuracy")

    st.subheader("Phishing Detection Comparison")
    metric_columns = [
        "Precision (phishing)",
        "Recall (phishing)",
        "F1-score (phishing)",
    ]

    st.bar_chart(
        results_df.set_index("Model")[metric_columns]
    )

    best_model_name = results_df.iloc[0]["Model"]
    st.success(
        f"Highest phishing F1-score in this experiment: {best_model_name}"
    )

    st.download_button(
        "⬇️ Download Comparison Results",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    st.caption(
        "The highest score on this random split does not guarantee the best "
        "performance on new domains. Further validation is required."
    )

else:
    st.subheader("Algorithms included")
    st.markdown("""
    - **Random Forest:** Combines predictions from multiple decision trees.
    - **Logistic Regression:** A linear classification baseline.
    - **Decision Tree:** Learns feature-based decision rules.
    - **Gradient Boosting:** Builds an ensemble sequentially to correct errors.
    """)

    st.warning(
        "Click 'Run Model Comparison' to train and evaluate all four models. "
        "This may take a few minutes."
    )
