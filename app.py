import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "hgb_accident_model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def train_model():
    data = pd.read_csv("sample_data.csv")  # sample صغير من الداتا

    X = data.drop("Severity", axis=1)
    y = data["Severity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = HistGradientBoostingClassifier(
        max_iter=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model, scaler = train_model()

st.title("🚦 US Accidents Severity Predictor")