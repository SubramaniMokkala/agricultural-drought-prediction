import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Drought Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌾 Agricultural Drought Prediction System 🌾</div>', 
            unsafe_allow_html=True)
st.markdown("### 📍 Region: Maharashtra, India | 🛰️ Data: Satellite-based (2015-2024)")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/random_forest_drought_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except:
    st.error("⚠️ Error loading model. Please ensure model files exist in 'models/' folder.")
    st.stop()

# Sidebar - Input Parameters
st.sidebar.header("🎛️ Input Parameters")
st.sidebar.markdown("---")

# Vegetation Health
st.sidebar.subheader("🌱 Vegetation Indicators")
ndvi = st.sidebar.slider(
    "NDVI (Normalized Difference Vegetation Index)",
    min_value=0.20, max_value=0.70, value=0.45, step=0.01,
    help="Higher values = healthier vegetation. Range: 0.20-0.70"
)

vci = st.sidebar.slider(
    "VCI (Vegetation Condition Index)",
    min_value=0.0, max_value=100.0, value=50.0, step=1.0,
    help="VCI < 35 indicates drought stress"
)

# Precipitation
st.sidebar.subheader("🌧️ Precipitation Data")
precip_current = st.sidebar.number_input(
    "Current Month Precipitation (mm)",
    min_value=0.0, max_value=500.0, value=50.0, step=5.0
)

precip_3month = st.sidebar.number_input(
    "3-Month Cumulative Precipitation (mm)",
    min_value=0.0, max_value=1000.0, value=150.0, step=10.0
)

precip_6month = st.sidebar.number_input(
    "6-Month Cumulative Precipitation (mm)",
    min_value=0.0, max_value=2000.0, value=400.0, step=20.0
)

# Temperature
st.sidebar.subheader("🌡️ Temperature")
temp_mean = st.sidebar.slider(
    "Mean Temperature (°C)",
    min_value=15.0, max_value=40.0, value=27.0, step=0.5
)

# Additional features
st.sidebar.subheader("📊 Additional Metrics")
ndvi_3month_avg = st.sidebar.slider(
    "3-Month Average NDVI",
    min_value=0.20, max_value=0.70, value=0.43, step=0.01
)

precip_3month_avg = precip_3month / 3
precip_anomaly = st.sidebar.slider(
    "Precipitation Anomaly (%)",
    min_value=-100.0, max_value=150.0, value=0.0, step=5.0
)

precip_lag1 = st.sidebar.number_input(
    "Previous Month Precipitation (mm)",
    min_value=0.0, max_value=500.0, value=40.0, step=5.0
)

ndvi_lag1 = st.sidebar.slider(
    "Previous Month NDVI",
    min_value=0.20, max_value=0.70, value=0.42, step=0.01
)

# Prepare input for prediction
input_data = pd.DataFrame({
    'ndvi': [ndvi],
    'precipitation_mm': [precip_current],
    'temp_mean_c': [temp_mean],
    'precip_3month': [precip_3month],
    'precip_6month': [precip_6month],
    'ndvi_3month_avg': [ndvi_3month_avg],
    'precip_3month_avg': [precip_3month_avg],
    'vci': [vci],
    'precip_anomaly': [precip_anomaly],
    'precip_lag1': [precip_lag1],
    'ndvi_lag1': [ndvi_lag1]
})

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

drought_categories = ['No Drought', 'Moderate Drought', 'Severe Drought']
drought_colors = ['#2ecc71', '#f39c12', '#e74c3c']

# Main content
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.markdown("### 🎯 Prediction Result")
    predicted_category = drought_categories[prediction]
    predicted_color = drought_colors[prediction]
    
    st.markdown(f"""
        <div style='background-color: {predicted_color}; padding: 30px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>{predicted_category}</h2>
            <p style='color: white; font-size: 18px; margin-top: 10px;'>
                Confidence: {prediction_proba[prediction]*100:.1f}%
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Probability Distribution")
    fig_proba = go.Figure(data=[
        go.Bar(
            x=drought_categories,
            y=prediction_proba * 100,
            marker_color=drought_colors,
            text=[f'{p*100:.1f}%' for p in prediction_proba],
            textposition='auto',
        )
    ])
    fig_proba.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Probability (%)",
        showlegend=False
    )
    st.plotly_chart(fig_proba, use_container_width=True)

with col3:
    st.markdown("### 🌡️ Risk Indicators")
    
    # Risk assessment
    risk_score = 0
    if vci < 35:
        risk_score += 3
    elif vci < 50:
        risk_score += 2
    
    if ndvi < 0.35:
        risk_score += 3
    elif ndvi < 0.45:
        risk_score += 2
    
    if precip_3month < 100:
        risk_score += 3
    elif precip_3month < 200:
        risk_score += 2
    
    risk_level = "🟢 Low" if risk_score <= 3 else "🟡 Medium" if risk_score <= 6 else "🔴 High"
    
    st.metric("Overall Risk Level", risk_level)
    st.metric("VCI Status", f"{vci:.1f}" + (" ⚠️" if vci < 35 else " ✅"))
    st.metric("NDVI Status", f"{ndvi:.2f}" + (" ⚠️" if ndvi < 0.35 else " ✅"))

# Detailed Analysis
st.markdown("---")
st.markdown("## 📈 Detailed Analysis")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🌱 Vegetation Health Trends")
    
    # Gauge chart for NDVI
    fig_ndvi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ndvi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "NDVI", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0.2, 0.7], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0.2, 0.35], 'color': "#e74c3c"},
                {'range': [0.35, 0.50], 'color': "#f39c12"},
                {'range': [0.50, 0.7], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.35
            }
        }
    ))
    fig_ndvi.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_ndvi, use_container_width=True)

with col_b:
    st.markdown("### 🌧️ Precipitation Analysis")
    
    precip_data = pd.DataFrame({
        'Period': ['Current Month', '3-Month Cumulative', '6-Month Cumulative'],
        'Precipitation (mm)': [precip_current, precip_3month, precip_6month],
        'Color': ['#3498db', '#2980b9', '#21618c']
    })
    
    fig_precip = px.bar(
        precip_data,
        x='Period',
        y='Precipitation (mm)',
        color='Period',
        color_discrete_sequence=['#3498db', '#2980b9', '#21618c'],
        text='Precipitation (mm)'
    )
    fig_precip.update_traces(texttemplate='%{text:.1f}mm', textposition='outside')
    fig_precip.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_precip, use_container_width=True)

# Feature Importance
st.markdown("---")
st.markdown("## 🔍 What Influenced This Prediction?")

feature_importance = {
    'VCI': 0.264,
    'NDVI': 0.181,
    '6-Month Precipitation': 0.133,
    '3-Month Avg Precipitation': 0.098,
    '3-Month Precipitation': 0.095,
    'Previous Month Precipitation': 0.092
}

fig_importance = go.Figure(go.Bar(
    x=list(feature_importance.values()),
    y=list(feature_importance.keys()),
    orientation='h',
    marker_color='#2ecc71',
    text=[f'{v*100:.1f}%' for v in feature_importance.values()],
    textposition='auto'
))
fig_importance.update_layout(
    title="Feature Importance (Random Forest Model)",
    xaxis_title="Importance Score",
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_importance, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p><strong>Agricultural Drought Prediction System</strong></p>
        <p>Built with Streamlit | Model: Random Forest (87.5% Accuracy)</p>
        <p>Data Source: Google Earth Engine (MODIS NDVI, CHIRPS Precipitation, ERA5 Temperature)</p>
        <p>Author: Subramani Mokkala | November 2025</p>
    </div>
""", unsafe_allow_html=True)