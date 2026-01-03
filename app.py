import streamlit as st
import pickle
import numpy as np

# Load the single pkl file
data = pickle.load(open('football_app.pkl', 'rb'))

models = data['models']
scaler = data['scaler']

st.title("⚽ Fantasy Football Clean Sheet Predictor")

model_name = st.selectbox(
    "Select Machine Learning Model",
    ("logistic_regression", "decision_tree", "random_forest")
)

model = models[model_name]

# Inputs
minutes = st.number_input("Minutes Played", 0, 120)
goals_conceded = st.number_input("Goals Conceded", 0, 10)
saves = st.number_input("Saves", 0, 20)
was_home = st.selectbox("Home Match?", [0, 1])
influence = st.number_input("Influence", 0.0)
creativity = st.number_input("Creativity", 0.0)
threat = st.number_input("Threat", 0.0)
ict_index = st.number_input("ICT Index", 0.0)
opponent_team = st.number_input("Opponent Team ID", 1)

if st.button("Predict Clean Sheet"):
    X = np.array([[minutes, goals_conceded, saves, was_home,
                   influence, creativity, threat, ict_index, opponent_team]])
    X = scaler.transform(X)
    result = model.predict(X)

    if result[0] == 1:
        st.success("✅ Clean Sheet Expected")
    else:
        st.error("❌ No Clean Sheet")
