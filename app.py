import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("accident_severity_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="US Accidents Severity Predictor", layout="centered")
st.title("🚦 US Accidents Severity Predictor")

st.sidebar.header("Enter Accident Details")

Hour = st.sidebar.slider("Hour", 0, 23, 8)
Weekday = st.sidebar.slider("Weekday (0=Mon)", 0, 6, 0)
Month = st.sidebar.slider("Month", 1, 12, 1)
Distance = st.sidebar.number_input("Distance (mi)", 0.0, 50.0, 0.5)
Temperature = st.sidebar.number_input("Temperature (F)", -50.0, 120.0, 65.0)
Wind_Chill = st.sidebar.number_input("Wind Chill (F)", -50.0, 120.0, 60.0)
Humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 70.0)
Pressure = st.sidebar.number_input("Pressure (in)", 28.0, 32.0, 29.9)
Visibility = st.sidebar.number_input("Visibility (mi)", 0.0, 10.0, 10.0)
Wind_Speed = st.sidebar.number_input("Wind Speed (mph)", 0.0, 100.0, 5.0)
Precipitation = st.sidebar.number_input("Precipitation (in)", 0.0, 5.0, 0.0)

Traffic_Signal = st.sidebar.checkbox("Traffic Signal")
Junction = st.sidebar.checkbox("Junction")
Crossing = st.sidebar.checkbox("Crossing")
Stop = st.sidebar.checkbox("Stop")
Railway = st.sidebar.checkbox("Railway")
Roundabout = st.sidebar.checkbox("Roundabout")

# Input DataFrame (⚠️ نفس ترتيب التدريب)
input_data = pd.DataFrame([[
    Hour, Weekday, Month, Distance, Temperature, Wind_Chill,
    Humidity, Pressure, Visibility, Wind_Speed, Precipitation,
    int(Traffic_Signal), int(Junction), int(Crossing),
    int(Stop), int(Railway), int(Roundabout)
]], columns=[
    'Hour', 'Weekday', 'Month', 'Distance(mi)', 'Temperature(F)',
    'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)', 'Traffic_Signal',
    'Junction', 'Crossing', 'Stop', 'Railway', 'Roundabout'
])

if st.button("Predict Severity"):
    try:
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled).max()

        st.success(f"🚨 Predicted Severity Level: **{pred[0]}**")
        st.info(f"Confidence: **{proba*100:.2f}%**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))
