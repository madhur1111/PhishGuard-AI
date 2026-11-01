
from pathlib import Path
from urllib.parse import urlparse
import ipaddress
import math
import re
from collections import Counter

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"

st.set_page_config(
    page_title="PhishGuard AI | Bulk Scanner",
    page_icon="🛡️",
    layout="wide",
)

st.title("📂 Bulk URL Scanner")
st.write(
    "Upload a CSV containing URLs to analyze multiple links in one operation."
)

st.info(
    "This tool provides machine-learning predictions, not a guarantee "
    "that a URL is safe or malicious."
)


# ---------- URL feature extraction ----------
def calculate_entropy(value):
    if not value:
        return 0.0

    counts = Counter(value)
    length = len(value)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def extract_features(raw_url, feature_columns):
    raw_url = str(raw_url).strip()

    if not raw_url:
        raise ValueError("URL is empty.")

    if not re.match(r"^https?://", raw_url, re.I):
        raw_url = "http://" + raw_url

    parsed = urlparse(raw_url)
    host = (parsed.hostname or "").lower()

    if not host:
        raise ValueError("Invalid URL or missing hostname.")

    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    parts = host.split(".")
    tld = parts[-1] if parts else ""
    query = parsed.query or ""
    path = parsed.path or ""
    digit_count = sum(character.isdigit() for character in raw_url)

    common_tlds = {
        "com", "org", "net", "edu", "gov", "in",
        "uk", "de", "io", "co", "info", "biz"
    }

    values = {
        "url_length": len(raw_url),
        "has_ip_address": has_ip,
        "dot_count": raw_url.count("."),
        "https_flag": int(parsed.scheme.lower() == "https"),
        "url_entropy": calculate_entropy(raw_url),
        "token_count": len(re.findall(r"[a-zA-Z0-9]+", raw_url)),
        "subdomain_count": max(0, len(parts) - 2),
        "query_param_count": (
            len([item for item in query.split("&") if item])
            if query else 0
        ),
        "tld_length": len(tld),
        "path_length": len(path),
        "has_hyphen_in_domain": int("-" in host),
        "number_of_digits": digit_count,
        "tld_popularity": int(tld in common_tlds),
        "suspicious_file_extension": int(
            bool(re.search(
                r"\.(exe|scr|bat|cmd|msi|dll|apk|zip)(?:$|[?#])",
                raw_url,
                re.I,
            ))
        ),
        "domain_name_length": len(host),
        "percentage_numeric_chars": (
            digit_count / len(raw_url) * 100 if raw_url else 0.0
        ),
    }

    return pd.DataFrame(
        [[values.get(column, 0) for column in feature_columns]],
        columns=feature_columns,
    )


# ---------- Load model ----------
if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. "
        "Run train_model.py before using this page."
    )
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ---------- CSV template ----------
template = pd.DataFrame({
    "URL": [
        "https://www.google.com",
        "https://github.com",
        "http://192.168.1.1/login.php",
    ]
})

st.download_button(
    "⬇️ Download Example CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="phishguard_example_urls.csv",
    mime="text/csv",
)

st.markdown("### Upload URLs")
st.write(
    "Your CSV must contain a column named `URL`. "
    "The example CSV above shows the required format."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")
        st.stop()

    input_df.columns = input_df.columns.str.strip()

    if "URL" not in input_df.columns:
        st.error(
            "A column named 'URL' was not found. "
            "Rename your URL column to URL and upload the file again."
        )
        st.stop()

    if input_df.empty:
        st.warning("The uploaded CSV contains no rows.")
        st.stop()

    if len(input_df) > 110000:
        st.error(
            "Please upload no more than 110000 rows per batch."
        )
        st.stop()

    st.write(f"**Rows uploaded:** {len(input_df):,}")
    st.dataframe(input_df.head(10), use_container_width=True)

    if st.button("🔍 Analyze All URLs", type="primary"):
        output_rows = []
        progress = st.progress(0)
        status = st.empty()

        for index, raw_url in enumerate(input_df["URL"], start=1):
            url_text = (
                str(raw_url).strip()
                if pd.notna(raw_url)
                else ""
            )

            result = {
                "URL": url_text,
                "Prediction": "Unable to analyze",
                "Model confidence (%)": None,
                "Status": "Error",
                "Error": "",
            }

            try:
                features = extract_features(
                    url_text,
                    feature_columns,
                )

                prediction = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                classes = list(model.classes_)
                confidence = float(
                    probabilities[classes.index(prediction)]
                ) * 100

                if prediction == 0:
                    result["Prediction"] = "Potentially phishing"
                elif prediction == 1:
                    result["Prediction"] = "Potentially legitimate"
                else:
                    result["Prediction"] = f"Class {prediction}"

                result["Model confidence (%)"] = round(confidence, 2)
                result["Status"] = "Analyzed"

            except Exception as error:
                result["Error"] = str(error)

            output_rows.append(result)

            if index % 10 == 0 or index == len(input_df):
                progress.progress(index / len(input_df))
                status.write(f"Analyzed {index:,} of {len(input_df):,} URLs")

        results_df = pd.DataFrame(output_rows)
        st.session_state["bulk_results"] = results_df

        status.success("Batch analysis completed!")

    results_df = st.session_state.get("bulk_results")

    if results_df is not None:
        st.markdown("---")
        st.subheader("📊 Analysis Summary")

        analyzed = results_df[results_df["Status"] == "Analyzed"]
        phishing_count = (
            analyzed["Prediction"] == "Potentially phishing"
        ).sum()
        legitimate_count = (
            analyzed["Prediction"] == "Potentially legitimate"
        ).sum()
        error_count = (results_df["Status"] == "Error").sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total URLs", f"{len(results_df):,}")
        col2.metric("Potentially phishing", int(phishing_count))
        col3.metric("Potentially legitimate", int(legitimate_count))
        col4.metric("Errors", int(error_count))

        if not analyzed.empty:
            chart_df = pd.DataFrame({
                "Category": [
                    "Potentially phishing",
                    "Potentially legitimate",
                ],
                "Count": [
                    int(phishing_count),
                    int(legitimate_count),
                ],
            }).set_index("Category")

            st.subheader("Prediction Distribution")
            st.bar_chart(chart_df)

        st.subheader("📋 Detailed Results")
        st.dataframe(results_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Analysis Report",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="phishguard_bulk_scan_results.csv",
            mime="text/csv",
        )

        st.caption(
            "Model confidence is not a calibrated probability of safety. "
            "A legitimate prediction does not guarantee a URL is safe."
        )
