import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fantasy Football Predictor",
    page_icon="⚽",
    layout="centered"
)

# -----------------------------
# Debug: show files (remove later if you want)
# -----------------------------
st.write("📂 Files in app directory:", os.listdir())

# -----------------------------
# Load model file safely
# -----------------------------
MODEL_FILE = "football_app.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("❌ football_app.pkl not found in repository")
    st.stop()

try:
    data = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"❌ Failed to load model file: {e}")
    st.stop()

# -----------------------------
# Extract models & scaler
# -----------------------------
models = data.get("models")
scaler = data.get("scaler")

if models is None:
    st.error("❌ Models not found inside pkl file")
    st.stop()

if scaler is None:
    st.warning("⚠️ Scaler not found. Predictions may be inaccurate.")

# -----------------------------
# App UI
# -----------------------------
st.title("⚽ Fantasy Football Clean Sheet Predictor")

model_name = st.selectbox(
    "Select Machine Learning Model",
    ["logistic_regression", "decision_tree", "random_forest"]
)

model = models[model_name]

st.subheader("Match & Player Statistics")

minutes = st.number_input("Minutes Played", min_value=0, max_value=120, value=90)
goals_conceded = st.number_input("Goals Conceded", min_value=0, max_value=10, value=0)
saves = st.number_input("Saves", min_value=0, max_value=20, value=0)
was_home = st.selectbox("Home Match?", [0, 1])
influence = st.number_input("Influence", min_value=0.0)
creativity = st.number_input("Creativity", min_value=0.0)
threat = st.number_input("Threat", min_value=0.0)
ict_index = st.number_input("ICT Index", min_value=0.0)
opponent_team = st.number_input("Opponent Team ID", min_value=1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Clean Sheet"):

    X = np.array([[
        minutes,
        goals_conceded,
        saves,
        was_home,
        influence,
        creativity,
        threat,
        ict_index,
        opponent_team
    ]])

    # Scale input if scaler exists
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"⚠️ Scaling failed: {e}")

    try:
        prediction = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(X)[0][1] * 100
        else:
            confidence = None

        if prediction == 1:
            msg = "✅ Clean Sheet Expected"
            if confidence is not None:
                msg += f" (Confidence: {confidence:.1f}%)"
            st.success(msg)
        else:
            msg = "❌ No Clean Sheet Expected"
            if confidence is not None:
                msg += f" (Confidence: {100 - confidence:.1f}%)"
            st.error(msg)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
