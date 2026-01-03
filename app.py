# app.py
import streamlit as st
import pickle
import numpy as np
import os

# ------------------------------
# Debug: Show files in the directory
# ------------------------------
st.write("Files in app directory:", os.listdir())

# ------------------------------
# Load pickle file safely
# ------------------------------
try:
    with open("football_app.pkl", "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    st.error("❌ football_app.pkl not found in repo!")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module: {e}. Check requirements.txt and runtime.txt!")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading pickle: {e}")
    st.stop()

# ------------------------------
# Extract models and scaler
# ------------------------------
models = data.get("models", None)
scaler = data.get("scaler", None)

if models is None:
    st.error("❌ No models found in the pickle file!")
    st.stop()

if scaler is None:
    st.warning("⚠️ No scaler found. Predictions may be inaccurate for Logistic Regression.")

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("⚽ Fantasy Football Clean Sheet Predictor")

model_name = st.selectbox(
    "Select Machine Learning Model",
    ("logistic_regression", "decision_tree", "random_forest")
)

model = models[model_name]

# Feature Inputs
minutes = st.number_input("Minutes Played", 0, 120, 90)
goals_conceded = st.number_input("Goals Conceded", 0, 10, 0)
saves = st.number_input("Saves", 0, 20, 0)
was_home = st.selectbox("Home Match?", [0, 1])
influence = st.number_input("Influence", 0.0, 100.0, 0.0)
creativity = st.number_input("Creativity", 0.0, 100.0, 0.0)
threat = st.number_input("Threat", 0.0, 100.0, 0.0)
ict_index = st.number_input("ICT Index", 0.0, 100.0, 0.0)
opponent_team = st.number_input("Opponent Team ID", 1, 100, 1)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Clean Sheet"):
    # Prepare input
    X = np.array([[minutes, goals_conceded, saves, was_home,
                   influence, creativity, threat, ict_index, opponent_team]])

    # Scale if scaler exists
    if scaler:
        try:
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"⚠️ Could not scale input: {e}")

    # Predict
    try:
        result = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0][1]

        if result[0] == 1:
            msg = "✅ Clean Sheet Expected"
            if proba:
                msg += f" (Confidence: {proba*100:.1f}%)"
            st.success(msg)
        else:
            msg = "❌ No Clean Sheet"
            if proba:
                msg += f" (Confidence: {(1-proba)*100:.1f}%)"
            st.error(msg)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
