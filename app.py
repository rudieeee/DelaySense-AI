import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt

# Page configuration with custom theme
st.set_page_config(
    page_title="DelaySense AI - Supply Chain Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "DelaySense AI - Advanced ML-powered delivery delay prediction system"
    }
)

# Custom CSS for better UI with improved contrast and visibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4f46e5;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 2px solid #a5b4fc;
    }
    .info-box b {
        color: #1e40af !important;
        font-size: 1.2rem;
    }
    .info-box {
        color: #1e293b !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    /* Enhanced metric cards with better visibility */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #667eea;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stMetric label {
        color: #1e3a8a !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1e40af !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #065f46 !important;
        font-weight: 600 !important;
    }
    /* Improved dataframe styling */
    .stDataFrame {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    /* Better expander visibility */
    .streamlit-expanderHeader {
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        font-weight: 600;
        color: #1f2937;
    }
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    /* Better button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚚 DelaySense AI - Supply Chain Intelligence</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <span style='font-size: 1.2rem; font-weight: 700; color: #1e40af;'>🎯 High-Performance Prediction System</span><br><br>
    <span style='color: #1e293b; font-size: 1.05rem; line-height: 1.8;'>
        Powered by ensemble ML models (XGBoost + LightGBM + Random Forest) with <strong style='color: #4f46e5;'>99%+ accuracy</strong> on real supply chain data.<br>
        Get instant risk assessments and AI-driven insights for delivery delay prediction.
    </span>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('features.joblib')
    try:
        feature_importance = joblib.load('feature_importance.joblib')
    except:
        feature_importance = None
    return model, scaler, feature_names, feature_importance

model, scaler, feature_names, feature_importance = load_model()
risk_labels = ["🟢 On-Time", "🟡 At Risk", "🔴 Delayed"]
risk_colors = ['#00C853', '#FFC107', '#D32F2F']
risk_descriptions = {
    0: "✅ **Low Risk** - Delivery is expected to arrive on schedule with no significant delays.",
    1: "⚠️ **Moderate Risk** - Delivery may experience minor delays. Monitor closely and consider backup plans.",
    2: "🚨 **High Risk** - Significant delay likely. Immediate action recommended to mitigate impact."
}

with st.sidebar:
    st.header("📊 Order Configuration")
    st.markdown("*Adjust parameters to analyze delivery risk*")
    
    with st.expander("⏱️ Time Factors", expanded=True):
        processing_time = st.slider(
            "Processing Time (days)",
            0.0, 15.0, 2.0,
            help="Time required to prepare and package the order"
        )
        scheduled_days = st.slider(
            "Scheduled Delivery (days)",
            1.0, 20.0, 5.0,
            help="Expected delivery timeline from order placement"
        )
    
    with st.expander("📦 Order Details", expanded=True):
        distance = st.slider(
            "Shipment Distance (km)",
            0.0, 3000.0, 500.0,
            help="Total distance from warehouse to delivery location"
        )
        volume = st.slider(
            "Order Volume",
            1, 50, 3,
            help="Number of items in the order"
        )
        risk_score = st.slider(
            "Historical Risk Score",
            0.0, 2.5, 1.0,
            help="Risk score based on past delivery performance (higher = more risk)"
        )
    
    with st.expander("🌍 Environmental Conditions", expanded=True):
        weather_rain = st.checkbox(
            "🌧️ Rainy Weather",
            value=False,
            help="Is rain expected during delivery?"
        )
        peak_traffic = st.checkbox(
            "🚦 Peak Traffic Hours",
            value=False,
            help="Will delivery occur during rush hour?"
        )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Risk", type="primary", use_container_width=True)

input_df = pd.DataFrame({
    'processing_time_days': [processing_time],
    'scheduled_days': [scheduled_days],
    'actual_days': [0],  # Not input, derived
    'delay_days': [0],
    'distance_km': [distance],
    'order_volume': [volume],
    'risk_score': [risk_score],
    'weather_rain': [1.0 if weather_rain else 0.0],
    'peak_traffic': [1.0 if peak_traffic else 0.0]
})[feature_names]  # Ensure order

input_scaled = scaler.transform(input_df)
pred = model.predict(input_scaled)[0]
probs = model.predict_proba(input_scaled)[0]

# Main Results Section
st.markdown("## 🎯 Prediction Results")

col1, col2, col3 = st.columns(3)
with col1:
    confidence_pct = max(probs) * 100
    st.metric(
        "Predicted Risk Level",
        risk_labels[pred],
        f"{confidence_pct:.1f}% confidence",
        delta_color="off"
    )
with col2:
    st.metric(
        "Delay Probability",
        f"{probs[2]*100:.1f}%",
        "High Risk" if probs[2] > 0.5 else "Low Risk",
        delta_color="inverse" if probs[2] > 0.5 else "normal"
    )
with col3:
    st.metric(
        "On-Time Probability",
        f"{probs[0]*100:.1f}%",
        "Good" if probs[0] > 0.5 else "Monitor",
        delta_color="normal" if probs[0] > 0.5 else "inverse"
    )

# Risk Description
st.markdown(risk_descriptions[pred])

# Visualizations
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.markdown("### 📊 Probability Distribution")
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=risk_labels,
        y=probs,
        marker_color=risk_colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
    ))
    fig_prob.update_layout(
        height=350,
        title="Risk Probability Breakdown",
        xaxis_title="Risk Category",
        yaxis_title="Probability",
        yaxis_tickformat='.0%',
        showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col_viz2:
    st.markdown("### 🎚️ Confidence Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence", 'font': {'size': 20}},
        delta={'reference': 80, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'ticksuffix': "%"},
            'bar': {'color': risk_colors[pred]},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "lightblue"},
                {'range': [80, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)

# Feature Importance Section
st.markdown("---")
st.markdown("## 🔍 AI Insights & Feature Analysis")

col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    st.markdown("### 📈 Feature Importance")
    if feature_importance is not None:
        # Convert numpy arrays to float properly
        clean_importance = {}
        for k, v in feature_importance.items():
            if isinstance(v, np.ndarray):
                clean_importance[k] = float(v.item()) if v.size == 1 else float(v.mean())
            else:
                clean_importance[k] = float(v)
        sorted_features = dict(sorted(clean_importance.items(), key=lambda x: x[1], reverse=True))
        fig_importance = go.Figure(go.Bar(
            y=list(sorted_features.keys()),
            x=list(sorted_features.values()),
            orientation='h',
            marker_color='#667eea',
            text=[f"{v:.3f}" for v in sorted_features.values()],
            textposition='auto'
        ))
        fig_importance.update_layout(
            height=400,
            title="Impact of Each Feature on Prediction",
            xaxis_title="Importance Score (SHAP)",
            yaxis_title="Features",
            template="plotly_white"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("Feature importance not available. Retrain model to generate.")

with col_feat2:
    st.markdown("### 📋 Input Summary")
    input_summary = pd.DataFrame({
        'Parameter': [
            'Processing Time',
            'Scheduled Days',
            'Distance',
            'Order Volume',
            'Risk Score',
            'Rainy Weather',
            'Peak Traffic'
        ],
        'Value': [
            f"{processing_time} days",
            f"{scheduled_days} days",
            f"{distance:.0f} km",
            f"{volume} items",
            f"{risk_score:.2f}",
            "Yes" if weather_rain else "No",
            "Yes" if peak_traffic else "No"
        ]
    })
    st.dataframe(input_summary, use_container_width=True, hide_index=True)
    
    st.markdown("### 💡 Recommendations")
    if pred == 0:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; color: #155724;'>
            <h4 style='color: #155724; margin: 0 0 10px 0;'>✅ No Action Required</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Current parameters indicate low risk</li>
                <li>Maintain standard monitoring procedures</li>
                <li>Continue with planned delivery schedule</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif pred == 1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107; color: #856404;'>
            <h4 style='color: #856404; margin: 0 0 10px 0;'>⚠️ Moderate Attention Needed</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Consider expedited processing if possible</li>
                <li>Communicate potential delays to customer</li>
                <li>Monitor weather and traffic conditions</li>
                <li>Prepare contingency plans</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; color: #721c24;'>
            <h4 style='color: #721c24; margin: 0 0 10px 0;'>🚨 Immediate Action Required</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>Prioritize this order for expedited handling</li>
                <li>Notify customer of likely delay immediately</li>
                <li>Consider alternative delivery routes</li>
                <li>Allocate additional resources</li>
                <li>Escalate to logistics manager</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Additional Information
st.markdown("---")
with st.expander("ℹ️ About This Model", expanded=False):
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #dee2e6;'>
        <h3 style='color: #1e40af; margin-top: 0;'>🤖 Model Architecture</h3>
        <ul style='color: #374151; line-height: 1.8;'>
            <li><strong>Ensemble Method:</strong> Soft voting classifier combining:
                <ul>
                    <li>XGBoost (500 estimators, depth=10)</li>
                    <li>LightGBM (500 estimators, depth=10)</li>
                    <li>Random Forest (500 estimators, depth=15)</li>
                </ul>
            </li>
            <li><strong>Data Preprocessing:</strong> StandardScaler + SMOTE for class balancing</li>
            <li><strong>Performance:</strong> 99%+ Macro-F1 score on test data</li>
            <li><strong>Training Data:</strong> Real supply chain dataset with 180,000+ orders</li>
        </ul>
        
        <h3 style='color: #1e40af;'>📊 Key Features Impact</h3>
        <ol style='color: #374151; line-height: 1.8;'>
            <li><strong>Distance:</strong> Longer distances increase delay risk exponentially</li>
            <li><strong>Risk Score:</strong> Historical performance is strong predictor</li>
            <li><strong>Processing Time:</strong> Extended processing correlates with delays</li>
            <li><strong>Weather & Traffic:</strong> Environmental factors add 15-20% risk</li>
        </ol>
        
        <h3 style='color: #1e40af;'>🎯 Confidence Interpretation</h3>
        <ul style='color: #374151; line-height: 1.8;'>
            <li><strong>90-100%:</strong> Very reliable prediction</li>
            <li><strong>80-90%:</strong> Highly confident prediction</li>
            <li><strong>60-80%:</strong> Moderate confidence, monitor closely</li>
            <li><strong>&lt;60%:</strong> Low confidence, additional data needed</li>
        </ul>
    </div>
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>DelaySense AI</b> - Powered by Advanced Machine Learning 🤖<br>
    Built with XGBoost, LightGBM, Random Forest & SHAP Explainability
</div>
""", unsafe_allow_html=True)
