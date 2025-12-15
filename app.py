import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('hgb_accident_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🚦 US Accidents Severity Predictor")

# Sidebar user input
st.sidebar.header("Enter Accident Details")
Hour = st.sidebar.slider("Hour", 0, 23, 8)
Weekday = st.sidebar.slider("Weekday (0=Mon)", 0, 6, 0)
Day = st.sidebar.slider("Day (1-31)", 1, 31, 1)  # العمود المفقود
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

# Build DataFrame
input_data = pd.DataFrame({
    'Hour':[Hour],
    'Weekday':[Weekday],
    'Day':[Day],  # مهم جدًا
    'Month':[Month],
    'Distance(mi)':[Distance],
    'Temperature(F)':[Temperature],
    'Wind_Chill(F)':[Wind_Chill],
    'Humidity(%)':[Humidity],
    'Pressure(in)':[Pressure],
    'Visibility(mi)':[Visibility],
    'Wind_Speed(mph)':[Wind_Speed],
    'Precipitation(in)':[Precipitation],
    'Traffic_Signal':[int(Traffic_Signal)],
    'Junction':[int(Junction)],
    'Crossing':[int(Crossing)],
    'Stop':[int(Stop)],
    'Railway':[int(Railway)],
    'Roundabout':[int(Roundabout)]
})

# Ensure columns order matches training
expected_columns = ['Hour', 'Weekday', 'Day', 'Month', 'Distance(mi)',
                    'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                    'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                    'Precipitation(in)', 'Traffic_Signal', 'Junction', 
                    'Crossing', 'Stop', 'Railway', 'Roundabout']
input_data = input_data[expected_columns]

# Prediction function
def predict_severity(df):
    try:
        df_scaled = scaler.transform(df)
        pred_class = model.predict(df_scaled)
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(df_scaled).max(axis=1)
        else:
            pred_proba = np.ones(len(pred_class))  # fallback
        return pred_class, pred_proba
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return ["Error"], [0]

# Make prediction
pred_class, pred_proba = predict_severity(input_data)

st.subheader("Predicted Accident Severity")
st.write(f"Severity Level: {pred_class[0]}")
st.write(f"Prediction Confidence: {pred_proba[0]*100:.2f}%")
