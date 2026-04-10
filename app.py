"""
Cold Storage Fruit Quality Classifier — Streamlit App
Predicts whether environmental conditions are Optimal or Suboptimal
for spoilage prevention in cold storage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Load artefacts ──────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    base = os.path.dirname(__file__)
    model        = joblib.load(os.path.join(base, "model.joblib"))
    scaler       = joblib.load(os.path.join(base, "scaler.joblib"))
    label_enc    = joblib.load(os.path.join(base, "label_encoder.joblib"))
    feature_cols = joblib.load(os.path.join(base, "feature_cols.joblib"))
    return model, scaler, label_enc, feature_cols

model, scaler, label_enc, feature_cols = load_artefacts()

NUMERIC_COLS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
FRUIT_OPTIONS = ["Banana", "Orange", "Pineapple", "Tomato"]

# ── Page config ─────────────────────────────────────────────
st.set_page_config(page_title="🧊 Cold Storage Classifier", page_icon="🧊", layout="centered")

st.title("🧊 Cold Storage Fruit Quality Classifier")
st.markdown(
    "Predict whether the cold-storage environment is **Optimal (Good)** or "
    "**Suboptimal (Bad)** for spoilage prevention based on sensor readings."
)

st.divider()

# ── Sidebar — About ────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app uses a **Random Forest** model trained on ~11 000 cold-storage "
        "sensor records. It classifies conditions as *Good* (optimal) or "
        "*Bad* (suboptimal) for fruit spoilage prevention."
    )
    st.markdown("**Features used:**")
    st.markdown("- Fruit type\n- Temperature (°C)\n- Humidity (%)\n- Light intensity (Fux)\n- CO₂ concentration (ppm)")
    st.divider()
    st.caption("Built with Scikit-learn & Streamlit")

# ── Input form ──────────────────────────────────────────────
st.subheader("Enter Sensor Readings")

col1, col2 = st.columns(2)
with col1:
    fruit = st.selectbox("Fruit Type", FRUIT_OPTIONS)
    temp  = st.slider("Temperature (°C)", min_value=18, max_value=30, value=24)
with col2:
    humid = st.slider("Humidity (%)", min_value=60, max_value=100, value=93)
    light = st.number_input("Light Intensity (Fux)", min_value=0.0, max_value=300.0, value=13.0, step=0.5)

co2 = st.slider("CO₂ Concentration (ppm)", min_value=0, max_value=500, value=320)

# ── Prediction ──────────────────────────────────────────────
if st.button("🔍 Predict Condition", type="primary", use_container_width=True):

    # Build a single-row dataframe matching the training schema
    row = {
        "Temp": temp,
        "Humid (%)": humid,
        "Light (Fux)": light,
        "CO2 (pmm)": co2,
        "Fruit_Banana": int(fruit == "Banana"),
        "Fruit_Orange": int(fruit == "Orange"),
        "Fruit_Pineapple": int(fruit == "Pineapple"),
        "Fruit_Tomato": int(fruit == "Tomato"),
    }
    X_input = pd.DataFrame([row])[feature_cols]

    # Scale numeric columns
    X_input[NUMERIC_COLS] = scaler.transform(X_input[NUMERIC_COLS])

    # Predict
    prediction = model.predict(X_input)[0]
    proba      = model.predict_proba(X_input)[0]
    label      = label_enc.inverse_transform([prediction])[0]

    st.divider()

    # Display result
    if label == "Good":
        st.success(f"✅ **Optimal Conditions (Good)**  —  Confidence: {proba[prediction]*100:.1f}%")
        st.balloons()
    else:
        st.error(f"⚠️ **Suboptimal Conditions (Bad)**  —  Confidence: {proba[prediction]*100:.1f}%")

    # Show probabilities
    st.markdown("**Class Probabilities:**")
    prob_df = pd.DataFrame({
        "Class": label_enc.classes_,
        "Probability": [f"{p*100:.1f}%" for p in proba]
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

# ── Batch prediction ────────────────────────────────────────
st.divider()
st.subheader("📁 Batch Prediction (Upload CSV)")
st.markdown("Upload a CSV with columns: `Fruit`, `Temp`, `Humid (%)`, `Light (Fux)`, `CO2 (pmm)`")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.dataframe(df_up.head(), use_container_width=True)

    if st.button("🚀 Run Batch Prediction", use_container_width=True):
        # Preprocess
        df_batch = pd.get_dummies(df_up, columns=["Fruit"], drop_first=False, dtype=int)
        # Ensure all fruit columns exist
        for c in feature_cols:
            if c not in df_batch.columns:
                df_batch[c] = 0
        df_batch = df_batch[feature_cols]
        df_batch[NUMERIC_COLS] = scaler.transform(df_batch[NUMERIC_COLS])

        preds = model.predict(df_batch)
        labels = label_enc.inverse_transform(preds)

        df_up["Predicted Class"] = labels
        st.dataframe(df_up, use_container_width=True)

        csv_out = df_up.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv", use_container_width=True)
