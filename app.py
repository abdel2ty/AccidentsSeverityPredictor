import streamlit as st
import pandas as pd
import joblib

# ======================
# Load model & scaler
# ======================
@st.cache_resource
def load_artifacts():
    model = joblib.load("hgb_accident_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Feature names used during training
FEATURE_NAMES = scaler.feature_names_in_

# ======================
# Prediction function
# ======================
def predict_severity(df: pd.DataFrame):
    # Ensure correct column order
    df = df[FEATURE_NAMES]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred_class = model.predict(df_scaled)
    pred_proba = model.predict_proba(df_scaled).max(axis=1)

    return pred_class, pred_proba


# ======================
# UI
# ======================
st.set_page_config(
    page_title="US Accidents Severity Predictor",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 US Accidents Severity Predictor")
st.markdown("Predict accident severity based on time, weather, and road conditions.")

# ======================
# Sidebar inputs
# ======================
st.sidebar.header("📝 Accident Details")

Hour = st.sidebar.slider("Hour", 0, 23, 8)
Weekday = st.sidebar.slider("Weekday (0 = Monday)", 0, 6, 0)
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

# ======================
# Build input dataframe
# ======================
input_data = pd.DataFrame({
    "Hour": [Hour],
    "Weekday": [Weekday],
    "Month": [Month],
    "Distance(mi)": [Distance],
    "Temperature(F)": [Temperature],
    "Wind_Chill(F)": [Wind_Chill],
    "Humidity(%)": [Humidity],
    "Pressure(in)": [Pressure],
    "Visibility(mi)": [Visibility],
    "Wind_Speed(mph)": [Wind_Speed],
    "Precipitation(in)": [Precipitation],
    "Traffic_Signal": [int(Traffic_Signal)],
    "Junction": [int(Junction)],
    "Crossing": [int(Crossing)],
    "Stop": [int(Stop)],
    "Railway": [int(Railway)],
    "Roundabout": [int(Roundabout)]
})

# ======================
# Prediction button
# ======================
if st.button("🚀 Predict Severity"):
    try:
        pred_class, pred_proba = predict_severity(input_data)

        st.subheader("📊 Prediction Result")

        st.metric(
            label="Predicted Severity Level",
            value=int(pred_class[0])
        )

        st.progress(float(pred_proba[0]))

        st.write(f"**Prediction Confidence:** {pred_proba[0]*100:.2f}%")

    except Exception as e:
        st.error("❌ Prediction failed. Please check input features.")
        st.exception(e)

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")
