from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
import ipaddress
import math
import re
import time

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"
DATA_PATH = BASE_DIR / "data" / "url_features_extracted1.csv"

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1120; color: #e5e7eb; }
.block-container { max-width: 1250px; padding-top: 1.6rem; }
.hero { padding: 26px; border-radius: 18px; background: linear-gradient(120deg,#13213b,#102c35); border: 1px solid #28415a; margin-bottom: 20px; }
.hero h1 { color: #5eead4; margin-bottom: 5px; }
.hero p { color: #cbd5e1; }
.small-note { color: #94a3b8; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

def entropy(value):
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())

def normalize_url(raw_url):
    value = str(raw_url).strip()
    if not value:
        raise ValueError("URL is empty.")
    if not re.match(r"^https?://", value, re.I):
        value = "http://" + value
    parsed = urlparse(value)
    if not parsed.hostname:
        raise ValueError("Could not identify a domain. Enter a URL such as https://example.com")
    return value, parsed

def extract_features(raw_url, feature_columns):
    raw_url, parsed = normalize_url(raw_url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0
    parts = host.split(".")
    tld = parts[-1] if parts else ""
    digits = sum(ch.isdigit() for ch in raw_url)
    common_tlds = {"com", "org", "net", "edu", "gov", "in", "uk", "de", "io", "co", "info", "biz"}
    values = {
        "url_length": len(raw_url),
        "has_ip_address": has_ip,
        "dot_count": raw_url.count("."),
        "https_flag": int(parsed.scheme.lower() == "https"),
        "url_entropy": entropy(raw_url),
        "token_count": len(re.findall(r"[a-zA-Z0-9]+", raw_url)),
        "subdomain_count": max(0, len(parts) - 2),
        "query_param_count": len([item for item in query.split("&") if item]) if query else 0,
        "tld_length": len(tld),
        "path_length": len(path),
        "has_hyphen_in_domain": int("-" in host),
        "number_of_digits": digits,
        "tld_popularity": int(tld in common_tlds),
        "suspicious_file_extension": int(bool(re.search(r"\.(exe|scr|bat|cmd|msi|dll|apk|zip)(?:$|[?#])", raw_url, re.I))),
        "domain_name_length": len(host),
        "percentage_numeric_chars": 100.0 * digits / len(raw_url) if raw_url else 0.0,
    }
    return pd.DataFrame([[values.get(name, 0) for name in feature_columns]], columns=feature_columns), raw_url, parsed, values

def predict_url(raw_url, model, feature_columns):
    features, normalized, parsed, values = extract_features(raw_url, feature_columns)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)]) * 100
    verdict = "Potentially phishing" if prediction == 0 else "Potentially legitimate" if prediction == 1 else f"Class {prediction}"
    return {"URL": str(raw_url).strip(), "Normalized URL": normalized, "Verdict": verdict, "Model confidence (%)": round(confidence, 2), "Prediction": prediction}, features, parsed, values

def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["URL", "ClassLabel"]).drop_duplicates()
    df["ClassLabel"] = pd.to_numeric(df["ClassLabel"], errors="coerce")
    df = df.dropna(subset=["ClassLabel"])
    df["ClassLabel"] = df["ClassLabel"].astype(int)
    feature_columns = [c for c in df.columns if c not in ["URL", "ClassLabel"] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_columns].copy().replace([float("inf"), float("-inf")], float("nan"))
    return X, df["ClassLabel"], feature_columns

if not MODEL_PATH.exists():
    st.error("Trained model not found. Place phishing_model.joblib beside app.py, or run train_model.py first.")
    st.stop()
try:
    artifact = load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None

st.markdown("""
<div class="hero">
<h1>🛡️ PhishGuard AI</h1>
<p>AI-powered URL analysis for identifying potentially malicious links.</p>
<span>Machine Learning · URL Intelligence · Bulk Analysis · Cybersecurity</span>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Security Overview", "URL Scanner", "Bulk Scanner", "URL Intelligence", "Model Lab", "Scan History", "About Project"])
st.sidebar.markdown("---")
st.sidebar.caption("PhishGuard AI | Academic Mini Project")

if page == "Security Overview":
    st.subheader("Security Overview")
    history_df = pd.DataFrame(st.session_state.scan_history)
    total = len(history_df)
    phishing = int((history_df["Verdict"] == "Potentially phishing").sum()) if total and "Verdict" in history_df else 0
    legitimate = int((history_df["Verdict"] == "Potentially legitimate").sum()) if total and "Verdict" in history_df else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Single URL scans", total)
    c2.metric("Potentially phishing", phishing)
    c3.metric("Potentially legitimate", legitimate)
    c4.metric("Model", "Random Forest")
    st.markdown("### Quick actions")
    st.write("Use the sidebar to scan a URL, analyze a CSV batch, inspect URL indicators, or compare models.")
    if total:
        st.markdown("### Recent scan activity")
        st.dataframe(history_df.head(10), use_container_width=True, hide_index=True)
        chart = history_df["Verdict"].value_counts().rename_axis("Verdict").to_frame("Scans")
        st.bar_chart(chart)
    else:
        st.info("No scans in this browser session yet. Start with URL Scanner or Bulk Scanner.")
    st.warning("Prototype note: statistics reflect this browser session only. Predictions are not a guarantee of real-world safety.")

elif page == "URL Scanner":
    st.subheader("🔍 URL Security Scanner")
    st.write("Enter a website address to estimate whether it resembles a phishing URL based on the trained dataset.")
    with st.form("single_scan_form"):
        url_input = st.text_input("Website URL", placeholder="https://example.com")
        submitted = st.form_submit_button("Analyze URL", type="primary", use_container_width=True)
    if submitted:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            try:
                result, features, parsed, values = predict_url(url_input, model, feature_columns)
                st.session_state.scan_history.insert(0, {k: result[k] for k in ["URL", "Verdict", "Model confidence (%)"]})
                st.session_state.scan_history = st.session_state.scan_history[:500]
                left, right = st.columns(2)
                with left:
                    if result["Prediction"] == 0:
                        st.error("⚠️ Potentially Phishing URL")
                    else:
                        st.success("✅ Potentially Legitimate URL")
                    st.metric("Model verdict", result["Verdict"])
                with right:
                    st.metric("Model confidence", f"{result['Model confidence (%)']:.2f}%")
                    st.progress(min(max(result["Model confidence (%)"] / 100, 0.0), 1.0))
                st.caption("Confidence is not a calibrated probability of real-world safety or maliciousness.")
                host = parsed.hostname or ""
                try:
                    ipaddress.ip_address(host)
                    has_ip = True
                except ValueError:
                    has_ip = False
                indicators = {
                    "Uses HTTPS": parsed.scheme.lower() == "https",
                    "Host is an IP address": has_ip,
                    "Long URL (over 75 characters)": len(result["Normalized URL"]) > 75,
                    "Contains @ symbol": "@" in result["Normalized URL"],
                    "Contains suspicious keywords": bool(re.search(r"login|verify|secure|account|update|password|wallet|confirm", result["Normalized URL"], re.I)),
                    "Contains punycode marker": "xn--" in result["Normalized URL"].lower(),
                    "Contains suspicious file extension": bool(values["suspicious_file_extension"]),
                }
                st.subheader("URL Indicators")
                st.dataframe(pd.DataFrame([{"Indicator": k, "Finding": "Yes" if v else "No"} for k, v in indicators.items()]), use_container_width=True, hide_index=True)
                st.subheader("Extracted Model Features")
                st.dataframe(features.T.rename(columns={0: "Value"}), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not analyze this URL: {exc}")
    st.caption("Educational prototype only. Do not use this tool as the sole basis for security decisions.")

elif page == "Bulk Scanner":
    st.subheader("📁 Bulk URL Scanner")
    st.write("Upload a CSV containing a column named `URL`. The example file is a template, not a threat verdict list.")
    template = pd.DataFrame({"URL": ["https://www.google.com", "https://github.com", "http://192.168.1.1/login.php"]})
    st.download_button("⬇️ Download Example CSV", template.to_csv(index=False).encode("utf-8"), "phishguard_example_urls.csv", "text/csv")
    uploaded = st.file_uploader("Upload URL CSV", type=["csv"], key="bulk_upload")
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            input_df.columns = input_df.columns.str.strip()
            if "URL" not in input_df.columns:
                st.error("CSV must contain a column named URL.")
            elif input_df.empty:
                st.warning("The uploaded CSV contains no rows.")
            elif len(input_df) > 10000:
                st.error("Please upload no more than 10,000 URLs per batch for a responsive demo.")
            else:
                st.write(f"Rows uploaded: {len(input_df):,}")
                st.dataframe(input_df.head(10), use_container_width=True)
                if st.button("🔍 Analyze All URLs", type="primary"):
                    rows = []
                    progress = st.progress(0)
                    status = st.empty()
                    for i, raw in enumerate(input_df["URL"], start=1):
                        url_text = str(raw).strip() if pd.notna(raw) else ""
                        try:
                            result, _, _, _ = predict_url(url_text, model, feature_columns)
                            rows.append({"URL": url_text, "Prediction": result["Verdict"], "Model confidence (%)": result["Model confidence (%)"], "Status": "Analyzed", "Error": ""})
                            st.session_state.scan_history.insert(0, {"URL": url_text, "Verdict": result["Verdict"], "Model confidence (%)": result["Model confidence (%)"]})
                        except Exception as exc:
                            rows.append({"URL": url_text, "Prediction": "Unable to analyze", "Model confidence (%)": None, "Status": "Error", "Error": str(exc)})
                        if i % 10 == 0 or i == len(input_df):
                            progress.progress(i / len(input_df))
                            status.write(f"Analyzed {i:,} of {len(input_df):,} URLs")
                    st.session_state.bulk_results = pd.DataFrame(rows)
                    st.session_state.scan_history = st.session_state.scan_history[:500]
                    status.success("Batch analysis completed!")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    results_df = st.session_state.bulk_results
    if results_df is not None:
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        analyzed = results_df[results_df["Status"] == "Analyzed"]
        phishing_count = int((analyzed["Prediction"] == "Potentially phishing").sum())
        legitimate_count = int((analyzed["Prediction"] == "Potentially legitimate").sum())
        errors = int((results_df["Status"] == "Error").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total URLs", len(results_df))
        c2.metric("Potentially phishing", phishing_count)
        c3.metric("Potentially legitimate", legitimate_count)
        c4.metric("Errors", errors)
        if not analyzed.empty:
            st.bar_chart(pd.DataFrame({"Count": [phishing_count, legitimate_count]}, index=["Potentially phishing", "Potentially legitimate"]))
        st.subheader("Detailed Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Analysis Report", results_df.to_csv(index=False).encode("utf-8"), "phishguard_bulk_scan_results.csv", "text/csv")
        st.caption("Predictions are model estimates, not confirmed threat intelligence.")

elif page == "URL Intelligence":
    st.subheader("🧠 URL Intelligence")
    st.write("Inspect structural indicators and extracted numerical features for a URL without treating any one indicator as proof of maliciousness.")
    intel_url = st.text_input("URL to inspect", placeholder="https://example.com", key="intel_url")
    if st.button("Inspect URL", type="primary"):
        try:
            result, features, parsed, values = predict_url(intel_url, model, feature_columns)
            host = parsed.hostname or ""
            try:
                ipaddress.ip_address(host)
                ip_host = True
            except ValueError:
                ip_host = False
            indicators = {
                "URL length > 75": len(result["Normalized URL"]) > 75,
                "HTTPS scheme": parsed.scheme.lower() == "https",
                "IP-address host": ip_host,
                "Hyphen in hostname": "-" in host,
                "Digits in URL": values["number_of_digits"] > 0,
                "Query parameters present": values["query_param_count"] > 0,
                "Suspicious file extension": values["suspicious_file_extension"] == 1,
                "Suspicious words present": bool(re.search(r"login|verify|secure|account|update|password|wallet|confirm", result["Normalized URL"], re.I)),
                "Punycode marker present": "xn--" in result["Normalized URL"].lower(),
            }
            a, b, c = st.columns(3)
            a.metric("Hostname", host or "Unknown")
            b.metric("URL length", len(result["Normalized URL"]))
            c.metric("Model verdict", result["Verdict"])
            st.subheader("Indicator checklist")
            st.dataframe(pd.DataFrame([{"Indicator": k, "Detected": "Yes" if v else "No"} for k, v in indicators.items()]), use_container_width=True, hide_index=True)
            st.subheader("Model feature values")
            st.dataframe(features.T.rename(columns={0: "Value"}), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not inspect URL: {exc}")

elif page == "Model Lab":
    st.subheader("🧪 Model Lab")
    st.write("Compare four algorithms on the supplied feature dataset using a random stratified holdout split.")
    st.info("Similar or related URLs may appear in both training and testing sets. These scores do not necessarily represent performance on unseen domains.")
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Keep data/url_features_extracted1.csv in the project folder.")
    else:
        try:
            X, y, data_features = prepare_data()
            c1, c2, c3 = st.columns(3)
            c1.metric("Dataset records", f"{len(X):,}")
            c2.metric("Numeric features", len(data_features))
            c3.metric("Class labels", y.nunique())
            if st.button("🚀 Run Model Comparison", type="primary"):
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.impute import SimpleImputer
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
                    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(random_state=42, class_weight="balanced")),
                    "Gradient Boosting": make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=42)),
                }
                results = []
                progress = st.progress(0)
                status = st.empty()
                for index, (name, candidate) in enumerate(models.items(), start=1):
                    status.write(f"Training {name}...")
                    start = time.time()
                    candidate.fit(X_train, y_train)
                    predictions = candidate.predict(X_test)
                    elapsed = time.time() - start
                    results.append({"Model": name, "Accuracy": accuracy_score(y_test, predictions), "Precision (phishing)": precision_score(y_test, predictions, pos_label=0, zero_division=0), "Recall (phishing)": recall_score(y_test, predictions, pos_label=0, zero_division=0), "F1-score (phishing)": f1_score(y_test, predictions, pos_label=0, zero_division=0), "Training + prediction time (seconds)": round(elapsed, 2)})
                    progress.progress(index / len(models))
                results_df = pd.DataFrame(results).sort_values("F1-score (phishing)", ascending=False)
                st.session_state.model_results = results_df
                status.success("All four models evaluated!")
            results_df = st.session_state.model_results
            if results_df is not None:
                st.subheader("Model Performance")
                st.dataframe(results_df.style.format({"Accuracy": "{:.4f}", "Precision (phishing)": "{:.4f}", "Recall (phishing)": "{:.4f}", "F1-score (phishing)": "{:.4f}"}), use_container_width=True, hide_index=True)
                st.subheader("Accuracy comparison")
                st.bar_chart(results_df.set_index("Model")[["Accuracy"]])
                st.subheader("Phishing detection comparison")
                st.bar_chart(results_df.set_index("Model")[["Precision (phishing)", "Recall (phishing)", "F1-score (phishing)"]])
                st.success(f"Highest phishing F1-score in this experiment: {results_df.iloc[0]['Model']}")
                st.download_button("⬇️ Download Comparison Results", results_df.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not prepare or compare models: {exc}")

elif page == "Scan History":
    st.subheader("📜 Scan History")
    history = st.session_state.scan_history
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Scan History CSV", history_df.to_csv(index=False).encode("utf-8"), "phishguard_scan_history.csv", "text/csv")
        if st.button("Clear History"):
            st.session_state.scan_history = []
            st.rerun()
    else:
        st.info("No scans yet. Go to URL Scanner or Bulk Scanner to begin.")
    st.caption("History is stored in the current browser session and may reset when the session ends.")

elif page == "About Project":
    st.subheader("📘 About PhishGuard AI")
    st.write("PhishGuard AI is an educational cybersecurity project that applies a trained machine-learning classifier to numerical URL characteristics to estimate whether a URL resembles a phishing link.")
    st.markdown("### Project objectives")
    st.markdown("- Extract URL characteristics.\n- Apply a trained classification model.\n- Display a verdict, confidence estimate, and structural indicators.\n- Analyze CSV batches.\n- Compare several machine-learning algorithms.")
    st.markdown("### Technology stack")
    st.markdown("Python · Streamlit · Pandas · Scikit-learn · Joblib")
    st.markdown("### Dataset and model")
    st.write("The main scanner loads the saved model artifact `phishing_model.joblib`. Model Lab reads `data/url_features_extracted1.csv` and evaluates four algorithms on a random holdout split.")
    st.warning("Important limitation: the URL feature extraction in this app is a practical approximation of the dataset features. Before making strong performance claims, verify that extraction matches the original dataset pipeline and test on unseen domains. A legitimate prediction does not guarantee a website is safe.")

st.markdown("---")
st.caption("PhishGuard AI · Academic Mini Project · Educational prototype only")
