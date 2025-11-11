import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Drought Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better spacing
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 30px;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        border-radius: 15px;
        margin-bottom: 40px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h3 {
        padding-top: 20px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌾 Agricultural Drought Prediction System</div>', 
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info("📍 **Region:** Maharashtra, India")
with col_info2:
    st.info("🛰️ **Data Source:** Satellite-based (2015-2024)")

st.markdown("<br><br>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/random_forest_drought_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except:
    st.error("⚠️ Error loading model. Please ensure model files exist in 'models/' folder.")
    st.stop()

# Sidebar - Input Parameters
st.sidebar.title("🎛️ Input Parameters")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Vegetation Health
with st.sidebar.expander("🌱 **Vegetation Indicators**", expanded=True):
    ndvi = st.slider(
        "NDVI",
        min_value=0.20, max_value=0.70, value=0.45, step=0.01,
        help="Normalized Difference Vegetation Index"
    )
    
    vci = st.slider(
        "VCI",
        min_value=0.0, max_value=100.0, value=50.0, step=1.0,
        help="Vegetation Condition Index"
    )
    
    ndvi_3month_avg = st.slider(
        "3-Month Avg NDVI",
        min_value=0.20, max_value=0.70, value=0.43, step=0.01
    )
    
    ndvi_lag1 = st.slider(
        "Previous Month NDVI",
        min_value=0.20, max_value=0.70, value=0.42, step=0.01
    )

# Precipitation
with st.sidebar.expander("🌧️ **Precipitation Data**", expanded=True):
    precip_current = st.number_input(
        "Current Month (mm)",
        min_value=0.0, max_value=500.0, value=50.0, step=5.0
    )
    
    precip_3month = st.number_input(
        "3-Month Cumulative (mm)",
        min_value=0.0, max_value=1000.0, value=150.0, step=10.0
    )
    
    precip_6month = st.number_input(
        "6-Month Cumulative (mm)",
        min_value=0.0, max_value=2000.0, value=400.0, step=20.0
    )
    
    precip_lag1 = st.number_input(
        "Previous Month (mm)",
        min_value=0.0, max_value=500.0, value=40.0, step=5.0
    )
    
    precip_anomaly = st.slider(
        "Precipitation Anomaly (%)",
        min_value=-100.0, max_value=150.0, value=0.0, step=5.0
    )

# Temperature
with st.sidebar.expander("🌡️ **Temperature**", expanded=False):
    temp_mean = st.slider(
        "Mean Temperature (°C)",
        min_value=15.0, max_value=40.0, value=27.0, step=0.5
    )

# Calculate derived features
precip_3month_avg = precip_3month / 3

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

# Main content - Prediction Result
st.markdown("## 🎯 Prediction Result")
st.markdown("<br>", unsafe_allow_html=True)

predicted_category = drought_categories[prediction]
predicted_color = drought_colors[prediction]

# Large prediction display
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, {predicted_color}DD 0%, {predicted_color} 100%); 
                    padding: 50px; border-radius: 20px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
            <h1 style='color: white; margin: 0; font-size: 48px;'>{predicted_category}</h1>
            <p style='color: white; font-size: 28px; margin-top: 20px; font-weight: bold;'>
                Confidence: {prediction_proba[prediction]*100:.1f}%
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Probability Distribution
st.markdown("## 📊 Probability Distribution")
st.markdown("<br>", unsafe_allow_html=True)

fig_proba = go.Figure(data=[
    go.Bar(
        x=drought_categories,
        y=prediction_proba * 100,
        marker_color=drought_colors,
        text=[f'{p*100:.1f}%' for p in prediction_proba],
        textposition='outside',
        textfont=dict(size=16, color='black', family='Arial Black')
    )
])
fig_proba.update_layout(
    height=400,
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis_title="Probability (%)",
    yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
    xaxis=dict(tickfont=dict(size=14)),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_proba, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Detailed Analysis
st.markdown("## 📈 Detailed Analysis")
st.markdown("<br>", unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("### 🌱 Vegetation Health (NDVI)")
    
    # Gauge chart for NDVI
    fig_ndvi = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ndvi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current NDVI", 'font': {'size': 24}},
        delta={'reference': 0.45, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0.2, 0.7], 'tickwidth': 2, 'tickfont': {'size': 14}},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'steps': [
                {'range': [0.2, 0.35], 'color': "#ffcccc"},
                {'range': [0.35, 0.50], 'color': "#fff4cc"},
                {'range': [0.50, 0.7], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.35
            }
        }
    ))
    fig_ndvi.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_ndvi, use_container_width=True)
    
    # VCI metric with custom styling
    vci_status = "Healthy ✅" if vci > 50 else "Stressed ⚠️"
    vci_color = "#2ecc71" if vci > 50 else "#e74c3c"
    st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 5px solid {vci_color}; margin-top: 20px;'>
            <p style='color: #7f8c8d; font-size: 14px; margin: 0;'>Vegetation Condition Index (VCI)</p>
            <h2 style='color: #2c3e50; margin: 5px 0;'>{vci:.1f}%</h2>
            <p style='color: {vci_color}; font-weight: bold; margin: 0;'>{vci_status}</p>
        </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("### 🌧️ Precipitation Analysis")
    
    precip_data = pd.DataFrame({
        'Period': ['Current\nMonth', '3-Month\nCumulative', '6-Month\nCumulative'],
        'Precipitation (mm)': [precip_current, precip_3month, precip_6month]
    })
    
    fig_precip = go.Figure(data=[
        go.Bar(
            x=precip_data['Period'],
            y=precip_data['Precipitation (mm)'],
            marker_color=['#3498db', '#2980b9', '#21618c'],
            text=precip_data['Precipitation (mm)'],
            texttemplate='%{text:.0f}mm',
            textposition='outside',
            textfont=dict(size=14, color='black')
        )
    ])
    fig_precip.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Precipitation (mm)",
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=12)),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_precip, use_container_width=True)
    
    # Risk indicators
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
    
    risk_level = "🟢 Low Risk" if risk_score <= 3 else "🟡 Medium Risk" if risk_score <= 6 else "🔴 High Risk"
    # Risk assessment with custom styling
    risk_color = "#2ecc71" if risk_score <= 3 else "#f39c12" if risk_score <= 6 else "#e74c3c"
    st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 5px solid {risk_color}; margin-top: 20px;'>
            <p style='color: #7f8c8d; font-size: 14px; margin: 0;'>Overall Risk Assessment</p>
            <h2 style='color: #2c3e50; margin: 5px 0;'>{risk_level}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 30px;'>
        <h4>Agricultural Drought Prediction System</h4>
        <p style='font-size: 14px;'>Built with Streamlit | Random Forest Model (87.5% Accuracy)</p>
        <p style='font-size: 13px;'>Data: Google Earth Engine (MODIS, CHIRPS, ERA5) | Author: Subramani Mokkala | 2025</p>
    </div>
""", unsafe_allow_html=True)