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

# Custom CSS 
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
        Powered by ensemble ML models (Logistic Regression + Decision Tree + Random Forest + XGBoost) with <strong style='color: #4f46e5;'>96.39% accuracy</strong> on real supply chain data.<br>
        Get instant risk assessments and AI-driven insights for delivery delay prediction.
    </span>
</div>
""", unsafe_allow_html=True)

# Quick Stats Section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; border: 2px solid #5568d3;'>
        <div style='font-size: 2.5rem; font-weight: bold;'>96.39%</div>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; border: 2px solid #e8477d;'>
        <div style='font-size: 2.5rem; font-weight: bold;'>4</div>
        <div style='font-size: 0.9rem; opacity: 0.9;'>AI Models</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; border: 2px solid #2b9fef;'>
        <div style='font-size: 2.5rem; font-weight: bold;'>180K+</div>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Training Samples</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; border: 2px solid #3be387;'>
        <div style='font-size: 2.5rem; font-weight: bold;'>0.97</div>
        <div style='font-size: 0.9rem; opacity: 0.9;'>Macro F1-Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
    st.markdown("*Configure delivery parameters for risk analysis*")
    
    with st.expander(" Order Timing", expanded=True):
        import datetime
        order_date = st.date_input(
            " Order Date",
            value=datetime.date.today(),
            help="When the order is placed"
        )
        
        scheduled_days = st.slider(
            "Scheduled Days",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Expected delivery timeline"
        )
        st.caption("💡 Recommended: 3-7 days for typical deliveries")
    
    with st.expander(" Shipment Details", expanded=True):
        distance = st.slider(
            "📍 Distance (km)",
            min_value=1.0,
            max_value=5000.0,
            value=500.0,
            step=10.0,
            help="Warehouse to delivery location distance"
        )
        
        km_per_day = distance / scheduled_days if scheduled_days > 0 else 0
        st.metric(
            "Speed Required", 
            f"{km_per_day:.0f} km/day",
            help="Distance divided by scheduled days"
        )
        
        volume = st.number_input(
            " Order Volume (items)",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            help="Number of items in the order"
        )
        
        processing_time = st.slider(
            " Processing Time (days)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.5,
            help="Time required for warehouse processing before shipment"
        )
        
        priority = st.select_slider(
            " Priority Level",
            options=["Low", "Standard", "High", "Express"],
            value="Standard",
            help="Delivery priority (for reference only)"
        )
    
    with st.expander("🌍 Environmental Conditions", expanded=True):
        weather_rain = st.checkbox(
            " Rainy Weather Expected",
            value=False,
            help="Rain conditions during delivery"
        )
        peak_traffic = st.checkbox(
            " Peak Traffic Hours",
            value=False,
            help="Delivery during rush hour"
        )
        
        
        st.markdown("---")
        st.caption("Additional Context (informational)")
        terrain = st.selectbox(
            " Terrain Type",
            ["Urban", "Suburban", "Rural", "Highway"],
            help="Route terrain (for reference)"
        )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Risk", type="primary", use_container_width=True)


import datetime
day_of_week = order_date.weekday()  # 0=Monday, 6=Sunday
is_weekend = 1 if day_of_week in [5, 6] else 0
month = order_date.month
is_holiday_season = 1 if month in [11, 12] else 0

# Calculate distance category 
if distance <= 100:
    distance_category = 0
elif distance <= 500:
    distance_category = 1
elif distance <= 1000:
    distance_category = 2
else:
    distance_category = 3

# Create base features
base_features = {
    'scheduled_days': scheduled_days,
    'distance_km': distance,
    'order_volume': float(volume),
    'processing_time': float(processing_time),
    'weather_rain': 1.0 if weather_rain else 0.0,
    'peak_traffic': 1.0 if peak_traffic else 0.0,
    'day_of_week': float(day_of_week),
    'is_weekend': float(is_weekend),
    'month': float(month),
    'is_holiday_season': float(is_holiday_season),
    'distance_category': float(distance_category)
}

# Create ALL interaction features (must match training)
weather_rain_val = 1.0 if weather_rain else 0.0
peak_traffic_val = 1.0 if peak_traffic else 0.0

interaction_features = {
    # Distance-based interactions
    'distance_weekend': distance * is_weekend,
    'distance_weather': distance * weather_rain_val,
    'distance_holiday': distance * is_holiday_season,
    'distance_traffic': distance * peak_traffic_val,
    'distance_scheduled': distance * scheduled_days,
    
    # Condition interactions
    'weekend_holiday': is_weekend * is_holiday_season,
    'weekend_weather': is_weekend * weather_rain_val,
    'weather_traffic': weather_rain_val * peak_traffic_val,
    'holiday_weather': is_holiday_season * weather_rain_val,
    
    # Volume-based interactions
    'volume_distance': volume * distance_category,
    'volume_weekend': volume * is_weekend,
    'volume_weather': volume * weather_rain_val,
    
    # Processing time interactions
    'processing_distance': processing_time * distance,
    'processing_volume': processing_time * volume,
    'processing_weekend': processing_time * is_weekend,
    
    # Triple interactions (compound effects)
    'distance_weekend_weather': distance * is_weekend * weather_rain_val,
    'distance_holiday_weather': distance * is_holiday_season * weather_rain_val,
    
    # Non-linear transformations
    'distance_squared': distance ** 2,
    'distance_log': np.log1p(distance),
    'scheduled_squared': scheduled_days ** 2,
    
    # Risk score (composite feature)
    'risk_score': (distance / 200) + (is_weekend * 3) + (weather_rain_val * 2.5) + (is_holiday_season * 4) + (processing_time * 0.5)
}

# Combine all features
all_features = {**base_features, **interaction_features}

# Create DataFrame with exact feature order from model
input_df = pd.DataFrame([all_features])[feature_names]

input_scaled = scaler.transform(input_df)
pred = model.predict(input_scaled)[0]
probs = model.predict_proba(input_scaled)[0]

# Main Results Section
st.markdown("🎯 Prediction Results")

# Calculate estimated delivery time for display 
base_delivery_days = (distance / 500.0)
weather_delay = 1.8 if weather_rain else 0.0
traffic_delay = 0.8 if peak_traffic else 0.0
weekend_delay = 2.5 if is_weekend else 0.0
holiday_delay = 3.0 if is_holiday_season else 0.0
processing_delay = processing_time * 0.5
estimated_actual_days = scheduled_days + base_delivery_days * 0.1 + weather_delay + traffic_delay + weekend_delay + holiday_delay + processing_delay
estimated_delay = max(0, estimated_actual_days - scheduled_days)

# Show delivery rationale
km_per_day = distance / scheduled_days if scheduled_days > 0 else 0


if km_per_day < 50:
    speed_assessment = "🟢 Very Reasonable - Plenty of time"
    speed_color = "#10b981"
elif km_per_day < 150:
    speed_assessment = "🟡 Moderate - Standard delivery pace"
    speed_color = "#f59e0b"
else:
    speed_assessment = "🔴 Tight - Fast delivery required"
    speed_color = "#ef4444"

# Build holiday and weekend indicators
holiday_indicator = '<span style="color: #dc2626; font-weight: bold;"> | 🎄 Holiday Season (+3 days risk)</span>' if is_holiday_season else ''
weekend_indicator = '<span style="color: #ea580c; font-weight: bold;"> | 📅 Weekend (+2.5 days risk)</span>' if is_weekend else ''

st.markdown(f"""
<div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b; margin-bottom: 20px; border: 2px solid #fbbf24;'>
    <div style='color: #92400e; font-weight: 600; font-size: 1.1rem; margin-bottom: 10px;'>📊 Delivery Configuration Analysis</div>
    <div style='color: #78350f; line-height: 1.8;'>
        <strong> Order Date:</strong> {order_date.strftime('%b %d, %Y')} ({['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][day_of_week]}){holiday_indicator}{weekend_indicator}<br>
        <strong> Journey:</strong> {distance:.0f} km in {scheduled_days:.1f} days = <span style='color: {speed_color}; font-weight: bold;'>{km_per_day:.1f} km/day</span><br>
        <strong>Speed Assessment:</strong> {speed_assessment}<br>
        <strong> Processing Time:</strong> {processing_time:.1f} days (warehouse prep)<br>
        <strong>Volume:</strong> {volume} item{'s' if volume > 1 else ''} | <strong>🌧️ Weather:</strong> {'Rain Expected (+1.8 days)' if weather_rain else 'Clear'} | <strong>🚦 Traffic:</strong> {'Peak Hours (+0.8 days)' if peak_traffic else 'Normal'}
    </div>
</div>
""", unsafe_allow_html=True)

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

# Add explanation box for counterintuitive predictions
if km_per_day < 50 and pred != 0:
    st.warning(f"""
    ** Why might this show risk despite having {km_per_day:.1f} km/day (plenty of time)?**
    
    The model learned from historical data patterns where very long scheduled times (>{scheduled_days:.0f} days) 
    sometimes correlate with:
    - Complex multi-stop deliveries
    - Special handling requirements
    - International shipments with customs delays
    - Large order volumes requiring careful processing
    
    **Recommendation:** If this is a simple direct delivery, the actual risk may be lower than predicted.
    Consider other factors beyond just distance/time ratio.
    """)

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
    st.markdown("### 📋 Configuration & Calculated Features")
    
    # Day of week names for display
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    distance_cat_names = ['Short (≤100km)', 'Medium (100-500km)', 'Long (500-1000km)', 'Very Long (>1000km)']
    
    input_summary = pd.DataFrame({
        'Feature': [
            '📅 Order Date',
            '📆 Day of Week',
            '📅 Weekend Order',
            '🎄 Holiday Season',
            '⏱️ Scheduled Days',
            '📍 Distance',
            '🗺️ Distance Category',
            '⚡ Required Speed',
            '📦 Order Volume',
            '🌧️ Rainy Weather',
            '🚦 Peak Traffic'
        ],
        'Value': [
            order_date.strftime('%B %d, %Y'),
            day_names[day_of_week],
            "✓ Yes" if is_weekend else "✗ No",
            "✓ Yes" if is_holiday_season else "✗ No",
            f"{scheduled_days:.1f} days",
            f"{distance:.0f} km",
            distance_cat_names[distance_category],
            f"{km_per_day:.1f} km/day",
            f"{volume} item{'s' if volume > 1 else ''}",
            "✓ Yes" if weather_rain else "✗ No",
            "✓ Yes" if peak_traffic else "✗ No"
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
        
        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px 0; color: white; text-align: center;'>
            <strong>📊 Ensemble Voting Classifier (Soft Voting)</strong>
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 15px 0;'>
            <div style='background: #e0e7ff; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea;'>
                <strong style='color: #3730a3;'>🔹 Logistic Regression</strong><br>
                <span style='font-size: 0.9rem; color: #4c1d95;'>Multinomial, L2 regularization, balanced weights</span>
            </div>
            <div style='background: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                <strong style='color: #92400e;'>🌳 Decision Tree</strong><br>
                <span style='font-size: 0.9rem; color: #78350f;'>Max depth 20, balanced class weights</span>
            </div>
            <div style='background: #d1fae5; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;'>
                <strong style='color: #065f46;'>🌲 Random Forest</strong><br>
                <span style='font-size: 0.9rem; color: #047857;'>100 estimators, max depth 20</span>
            </div>
            <div style='background: #ddd6fe; padding: 15px; border-radius: 8px; border-left: 4px solid #8b5cf6;'>
                <strong style='color: #5b21b6;'>⚡ XGBoost</strong><br>
                <span style='font-size: 0.9rem; color: #6b21a8;'>100 estimators, gradient boosting</span>
            </div>
        </div>
        
        <ul style='color: #374151; line-height: 1.8; margin-top: 20px;'>
            <li><strong>Voting Strategy:</strong> Soft voting with weights [3, 1, 3, 3] (higher weight for Random Forest and XGBoost)</li>
            <li><strong>Data Preprocessing:</strong> StandardScaler + SMOTE for class balancing</li>
            <li><strong>Performance:</strong> 96.39% Accuracy, 0.9658 Macro-F1 score</li>
            <li><strong>Training Data:</strong> Real supply chain dataset with 180,000+ orders</li>
            <li><strong>No Data Leakage:</strong> Only uses features available before delivery</li>
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

# ===== BATCH PREDICTION FEATURE =====
st.markdown("## 📊 Batch Prediction")
st.markdown("Upload a CSV file with multiple orders to get bulk predictions")

with st.expander("📤 Upload CSV for Batch Predictions", expanded=False):
    st.markdown("""
    **Required CSV Columns:**
    - `scheduled_days` - Expected delivery timeline (days)
    - `distance_km` - Distance in kilometers
    - `order_volume` - Number of items
    - `processing_time` - Warehouse processing time (days)
    - `weather_rain` - 1 if rain expected, 0 otherwise
    - `peak_traffic` - 1 if peak traffic, 0 otherwise
    - `day_of_week` - 0 (Monday) to 6 (Sunday)
    - `is_weekend` - 1 if weekend, 0 otherwise
    - `month` - Month number (1-12)
    - `is_holiday_season` - 1 if November/December, 0 otherwise
    - `distance_category` - 0 (≤100km), 1 (100-500km), 2 (500-1000km), 3 (>1000km)
    """)
    
    # Sample CSV template
    sample_data = {
        'scheduled_days': [5.0, 7.0, 3.0],
        'distance_km': [500.0, 1200.0, 150.0],
        'order_volume': [3, 5, 1],
        'processing_time': [1.5, 2.0, 1.0],
        'weather_rain': [0, 1, 0],
        'peak_traffic': [0, 1, 0],
        'day_of_week': [1, 5, 2],
        'is_weekend': [0, 0, 0],
        'month': [6, 12, 3],
        'is_holiday_season': [0, 1, 0],
        'distance_category': [1, 3, 1]
    }
    sample_csv = pd.DataFrame(sample_data).to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name="batch_template.csv",
        mime="text/csv",
        help="Download a sample CSV file with the correct format"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} orders")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔮 Predict All"):
                with st.spinner("Processing predictions..."):
                    # Create all interaction features for batch data
                    batch_processed = batch_df.copy()
                    
                    # Add interaction features
                    batch_processed['distance_weekend'] = batch_processed['distance_km'] * batch_processed['is_weekend']
                    batch_processed['distance_weather'] = batch_processed['distance_km'] * batch_processed['weather_rain']
                    batch_processed['distance_holiday'] = batch_processed['distance_km'] * batch_processed['is_holiday_season']
                    batch_processed['distance_traffic'] = batch_processed['distance_km'] * batch_processed['peak_traffic']
                    batch_processed['distance_scheduled'] = batch_processed['distance_km'] * batch_processed['scheduled_days']
                    batch_processed['weekend_holiday'] = batch_processed['is_weekend'] * batch_processed['is_holiday_season']
                    batch_processed['weekend_weather'] = batch_processed['is_weekend'] * batch_processed['weather_rain']
                    batch_processed['weather_traffic'] = batch_processed['weather_rain'] * batch_processed['peak_traffic']
                    batch_processed['holiday_weather'] = batch_processed['is_holiday_season'] * batch_processed['weather_rain']
                    batch_processed['volume_distance'] = batch_processed['order_volume'] * batch_processed['distance_category']
                    batch_processed['volume_weekend'] = batch_processed['order_volume'] * batch_processed['is_weekend']
                    batch_processed['volume_weather'] = batch_processed['order_volume'] * batch_processed['weather_rain']
                    batch_processed['processing_distance'] = batch_processed['processing_time'] * batch_processed['distance_km']
                    batch_processed['processing_volume'] = batch_processed['processing_time'] * batch_processed['order_volume']
                    batch_processed['processing_weekend'] = batch_processed['processing_time'] * batch_processed['is_weekend']
                    batch_processed['distance_weekend_weather'] = batch_processed['distance_km'] * batch_processed['is_weekend'] * batch_processed['weather_rain']
                    batch_processed['distance_holiday_weather'] = batch_processed['distance_km'] * batch_processed['is_holiday_season'] * batch_processed['weather_rain']
                    batch_processed['distance_squared'] = batch_processed['distance_km'] ** 2
                    batch_processed['distance_log'] = np.log1p(batch_processed['distance_km'])
                    batch_processed['scheduled_squared'] = batch_processed['scheduled_days'] ** 2
                    batch_processed['risk_score'] = (batch_processed['distance_km'] / 200) + (batch_processed['is_weekend'] * 3) + (batch_processed['weather_rain'] * 2.5) + (batch_processed['is_holiday_season'] * 4) + (batch_processed['processing_time'] * 0.5)
                    
                    # Align with model features
                    batch_input = batch_processed[feature_names]
                    batch_scaled = scaler.transform(batch_input)
                    
                    # Predictions
                    predictions = model.predict(batch_scaled)
                    probabilities = model.predict_proba(batch_scaled)
                    
                    # Add results to dataframe
                    results_df = batch_df.copy()
                    results_df['Predicted_Risk'] = [risk_labels[p] for p in predictions]
                    results_df['On-Time_Prob'] = probabilities[:, 0]
                    results_df['At_Risk_Prob'] = probabilities[:, 1]
                    results_df['Delayed_Prob'] = probabilities[:, 2]
                    results_df['Confidence'] = probabilities.max(axis=1)
                    
                    st.success("✅ Predictions Complete!")
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Orders", len(results_df))
                    with col2:
                        on_time_count = (predictions == 0).sum()
                        st.metric("🟢 On-Time", on_time_count, f"{on_time_count/len(results_df)*100:.1f}%")
                    with col3:
                        at_risk_count = (predictions == 1).sum()
                        st.metric("🟡 At Risk", at_risk_count, f"{at_risk_count/len(results_df)*100:.1f}%")
                    with col4:
                        delayed_count = (predictions == 2).sum()
                        st.metric("🔴 Delayed", delayed_count, f"{delayed_count/len(results_df)*100:.1f}%")
                    
                    # Display results
                    st.markdown("### 📋 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="delaysense_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    fig_dist = px.pie(
                        values=[on_time_count, at_risk_count, delayed_count],
                        names=risk_labels,
                        title="Risk Distribution",
                        color_discrete_sequence=risk_colors
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has all required columns with correct names")

# ===== EXPORT CURRENT PREDICTION =====
st.markdown("---")
st.markdown("## 💾 Export Current Prediction")

if 'pred' in locals():
    export_data = {
        'Order_Date': [order_date.strftime('%Y-%m-%d')],
        'Scheduled_Days': [scheduled_days],
        'Distance_KM': [distance],
        'Order_Volume': [volume],
        'Processing_Time': [processing_time],
        'Weather_Rain': [weather_rain],
        'Peak_Traffic': [peak_traffic],
        'Predicted_Risk': [risk_labels[pred]],
        'On-Time_Probability': [f"{probs[0]*100:.2f}%"],
        'At_Risk_Probability': [f"{probs[1]*100:.2f}%"],
        'Delayed_Probability': [f"{probs[2]*100:.2f}%"],
        'Confidence': [f"{max(probs)*100:.2f}%"]
    }
    export_df = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    with col1:
        csv_export = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_export,
            file_name=f"prediction_{order_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col2:
        json_export = export_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_export,
            file_name=f"prediction_{order_date.strftime('%Y%m%d')}.json",
            mime="application/json"
        )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>DelaySense AI</b> - Powered by Advanced Machine Learning 🤖<br>
    Built with Logistic Regression, Decision Tree, Random Forest, XGBoost & SHAP Explainability
</div>
""", unsafe_allow_html=True)
