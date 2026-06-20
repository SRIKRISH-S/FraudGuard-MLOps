import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
from datetime import datetime

from src.predict import FraudPredictor
from src.monitoring import DriftMonitor
from src.retrain import AutoRetrainer

# Page Config
st.set_page_config(
    page_title="FraudGuard MLOps Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #1e222b;
        border: 1px solid #2e3440;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
    }
    .header-style {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8585 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load local model info
@st.cache_resource
def load_predictor():
    try:
        return FraudPredictor()
    except Exception as e:
        return None

predictor = load_predictor()

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center;'>🛡️ FraudGuard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

# API and MLflow status check
api_url = os.getenv("API_URL", "http://localhost:8000")
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("FastAPI Service: ONLINE")
    else:
        st.sidebar.warning("FastAPI Service: UNHEALTHY")
except Exception:
    st.sidebar.warning("FastAPI Service: OFFLINE (Using local fallback)")

# Model metadata check
model_info = {}
if predictor:
    model_info = predictor.get_model_info()
    st.sidebar.info(f"Model Type: {model_info.get('model_type', 'Unknown')}\n\nROC-AUC: {model_info.get('roc_auc', 0.0):.4f}\n\nF1-Score: {model_info.get('f1_score', 0.0):.4f}")
else:
    st.sidebar.error("Model state: NOT LOADED. Train model first.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Title Header
st.markdown("<h1 class='header-style'>FraudGuard: Real-Time Fraud Detection & MLOps</h1>", unsafe_allow_html=True)
st.markdown("##### Production-Grade Real-Time Inferencing, Data Drift Monitoring, and Retraining Pipeline")
st.write("")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Predictor", 
    "📊 Batch Inference", 
    "📈 Drift & Monitoring", 
    "📜 Inference Audit Logs"
])

# ----------------- TAB 1: REAL-TIME PREDICTOR -----------------
with tab1:
    st.markdown("### Real-Time Transaction Verification")
    
    # Presets definition
    presets = {
        "Clean Transaction Profile": {
            "Time": 0.0, "Amount": 15.0,
            "V1": 1.19, "V2": 0.26, "V3": 0.16, "V4": 0.44, "V5": 0.06, "V6": -0.08, "V7": -0.07, "V8": 0.08,
            "V9": -0.25, "V10": -0.16, "V11": 1.61, "V12": 1.06, "V13": 0.48, "V14": -0.14, "V15": 0.63,
            "V16": 0.46, "V17": -0.11, "V18": -0.18, "V19": -0.14, "V20": -0.06, "V21": -0.22, "V22": -0.63,
            "V23": 0.10, "V24": -0.33, "V25": 0.16, "V26": 0.12, "V27": -0.01, "V28": 0.01
        },
        "High-Risk Fraud Profile": {
            "Time": 406.0, "Amount": 239.0,
            "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52, "V6": -1.42, "V7": -2.53, "V8": 1.39,
            "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -4.09, "V13": -0.19, "V14": -4.68, "V15": -0.12,
            "V16": -2.99, "V17": -4.61, "V18": -1.46, "V19": 0.42, "V20": 0.12, "V21": 0.51, "V22": -0.03,
            "V23": -0.46, "V24": 0.38, "V25": 0.04, "V26": 0.10, "V27": 0.35, "V28": 0.15
        }
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Select Profile Preset")
        selected_preset = st.selectbox("Choose a preset to fill form values:", list(presets.keys()))
        preset_values = presets[selected_preset]
        
        st.markdown("#### Transaction Details")
        tx_time = st.number_input("Time (Seconds elapsed)", value=float(preset_values["Time"]))
        tx_amount = st.number_input("Amount ($)", value=float(preset_values["Amount"]), min_value=0.0)
        
        # Collapse PCA inputs to save screen space
        with st.expander("Adjust PCA Encoded Features (V1-V28)"):
            pca_inputs = {}
            for i in range(1, 29):
                pca_inputs[f"V{i}"] = st.number_input(f"V{i}", value=float(preset_values[f"V{i}"]), format="%.6f")
                
        run_prediction = st.button("Analyze Transaction", use_container_width=True)

    with col2:
        st.markdown("#### Risk Analysis Report")
        if run_prediction:
            if predictor is None:
                st.error("Cannot run prediction: ML model artifacts not initialized on system.")
            else:
                # Build request payload
                payload = {"Time": tx_time, "Amount": tx_amount}
                payload.update(pca_inputs)
                
                # Make prediction
                with st.spinner("Analyzing transaction behavior..."):
                    results = predictor.predict(pd.DataFrame([payload]))
                    result = results[0]
                
                # Display Results
                prob = result["probability"]
                label = result["prediction"]
                risk_score = result["risk_score"]
                
                # Color code
                if label == 1:
                    st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
                    risk_color = "#ff4b4b"
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                    risk_color = "#00cc66"
                    
                # Gauge representation of risk
                st.markdown(f"""
                <div class='card' style='text-align: center; border-left: 8px solid {risk_color};'>
                    <div style='font-size: 1.1rem; color: #8892b0;'>FRAUD RISK SCORE</div>
                    <div style='font-size: 4rem; font-weight: 900; color: {risk_color};'>{risk_score} / 100</div>
                    <div style='font-size: 1rem; margin-top: 10px; color: #fafafa;'>
                        Probability: {prob:.4%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanatory chart
                fig, ax = plt.subplots(figsize=(6, 2))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1e222b')
                
                ax.barh(["Risk Index"], [risk_score], color=risk_color, height=0.4)
                ax.set_xlim(0, 100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#8892b0')
                ax.spines['left'].set_color('#8892b0')
                ax.tick_params(colors='#8892b0')
                ax.set_title("Inference Risk Spectrum", color='#fafafa')
                st.pyplot(fig)
        else:
            st.info("Click 'Analyze Transaction' to evaluate fraud probability.")

# ----------------- TAB 2: BATCH INFERENCE -----------------
with tab2:
    st.markdown("### Batch Transactions Processing")
    st.write("Upload a CSV file containing transactions matching the features schema (Time, Amount, V1-V28).")
    
    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV containing {len(df_upload)} transactions.")
            
            # Predict
            if predictor is None:
                st.error("ML model not loaded.")
            else:
                with st.spinner("Processing batch predictions..."):
                    results = predictor.predict(df_upload)
                    
                df_res = df_upload.copy()
                df_res["Prediction"] = [r["prediction"] for r in results]
                df_res["Probability"] = [r["probability"] for r in results]
                df_res["Risk_Score"] = [r["risk_score"] for r in results]
                
                fraud_count = df_res["Prediction"].sum()
                fraud_rate = fraud_count / len(df_res)
                
                # Metric Cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class='card'>
                        <div class='metric-label'>Total Transactions</div>
                        <div class='metric-value'>{len(df_res)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class='card'>
                        <div class='metric-label'>Fraud Flagged</div>
                        <div class='metric-value' style='color: #ff4b4b;'>{fraud_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class='card'>
                        <div class='metric-label'>Fraud Rate</div>
                        <div class='metric-value' style='color: #ffcc00;'>{fraud_rate:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show results table
                st.markdown("#### Processed Transaction Details")
                st.dataframe(
                    df_res[["Time", "Amount", "Prediction", "Probability", "Risk_Score"]].head(100),
                    use_container_width=True
                )
                
                # Visualizations
                st.markdown("#### Batch Visualization Summary")
                cv1, cv2 = st.columns(2)
                with cv1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#1e222b')
                    sns.histplot(df_res["Risk_Score"], bins=20, kde=True, color="salmon", ax=ax)
                    ax.set_title("Distribution of Transaction Risk Scores", color="white")
                    ax.set_xlabel("Risk Score", color="white")
                    ax.set_ylabel("Count", color="white")
                    ax.tick_params(colors="white")
                    st.pyplot(fig)
                with cv2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#1e222b')
                    labels = ["Legitimate", "Fraudulent"]
                    sizes = [len(df_res) - fraud_count, fraud_count]
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=["#00cc66", "#ff4b4b"], textprops={'color':"w"})
                    ax.set_title("Batch Prediction Ratios", color="white")
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ----------------- TAB 3: DRIFT & MONITORING -----------------
with tab3:
    st.markdown("### Production Monitoring & Data Drift")
    st.write("Compare production transaction batches against reference training schemas to detect drift.")
    
    # Active monitoring status
    drift_report_path = "logs/drift_report.html"
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("#### Trigger Automated Drift Check")
        st.write("Evaluate current inference logs for distribution shifts.")
        
        # Build dummy production data to test/demonstrate drift check
        if st.button("Run Live Drift Analysis"):
            train_path = "data/processed/train.csv"
            if not os.path.exists(train_path):
                st.error("Reference training data not found. Train model first.")
            else:
                with st.spinner("Analyzing dataset distributions..."):
                    try:
                        # Simulate production batch from test split
                        test_df = pd.read_csv("data/processed/test.csv")
                        # Add a small skew to simulate slight drift
                        prod_batch = test_df.sample(200, random_state=42).copy()
                        prod_batch["Amount"] = prod_batch["Amount"] * 1.5 + 10.0
                        
                        monitor = DriftMonitor()
                        drifted, summary = monitor.run_drift_check(
                            prod_batch,
                            report_html_path=drift_report_path
                        )
                        
                        if drifted:
                            st.warning("⚠️ DATA DRIFT DETECTED: Significant statistical shifts detected in features.")
                        else:
                            st.success("✅ NO DRIFT DETECTED: Feature distributions match training baseline.")
                            
                        st.json(summary)
                    except Exception as e:
                        st.error(f"Drift run failed: {e}")
                        
        st.markdown("#### Force Automated Model Retraining")
        st.write("Re-aggregate baseline & feedback data, re-run evaluation, and update MLflow model.")
        if st.button("Trigger Pipeline Retrain"):
            test_path = "data/processed/test.csv"
            if not os.path.exists(test_path):
                st.error("No training splits exist yet.")
            else:
                with st.spinner("Running retraining loops in MLflow..."):
                    try:
                        test_df = pd.read_csv(test_path)
                        retrainer = AutoRetrainer()
                        # Force retrain
                        success = retrainer.trigger_retrain(test_df.sample(100), force=True)
                        if success:
                            st.success("Automated retraining finished. Registered new champion model!")
                            st.cache_resource.clear()  # Clear cache to reload predictor
                        else:
                            st.info("Retraining script completed without modifying active model.")
                    except Exception as e:
                        st.error(f"Retraining pipeline failed: {e}")

    with c_m2:
        st.markdown("#### Data Drift Analysis Report")
        if os.path.exists(drift_report_path):
            st.caption("Rendering Evidently AI Drift Dashboard...")
            with open(drift_report_path, "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=500, scrolling=True)
        else:
            st.info("No Evidently drift reports generated yet. Click 'Run Live Drift Analysis' to produce a report.")

# ----------------- TAB 4: AUDIT LOGS -----------------
with tab4:
    st.markdown("### Inference Audit History")
    st.write("Chronological logs of transaction inquiries evaluated by the API/dashboard services.")
    
    audit_file = "logs/prediction_audit.csv"
    if os.path.exists(audit_file):
        try:
            df_audit = pd.read_csv(audit_file)
            
            # Sort by latest
            df_audit = df_audit.sort_values(by="timestamp", ascending=False)
            
            # Summary stats
            total_logged = len(df_audit)
            frauds_logged = df_audit["prediction"].sum()
            audit_rate = frauds_logged / total_logged if total_logged > 0 else 0.0
            
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.metric("Total API Requests", total_logged)
            with col_l2:
                st.metric("Fraud Inquiries Flagged", frauds_logged, delta=f"{audit_rate:.2%} rate", delta_color="inverse")
            with col_l3:
                st.metric("Peak Risk score logged", f"{df_audit['risk_score'].max()}%" if total_logged > 0 else "0.0%")
            
            st.markdown("#### Recent Predictions")
            st.dataframe(df_audit, use_container_width=True)
            
            # Clear logs button
            if st.button("Clear Prediction Audit Logs"):
                os.remove(audit_file)
                st.success("Audit log successfully reset.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to read audit log file: {e}")
    else:
        st.info("No predictions logged to audit log database yet. Run some predictions first!")
