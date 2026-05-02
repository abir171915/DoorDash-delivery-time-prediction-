import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(
    page_title="DoorDash Delivery Predictor",
    page_icon="🍕",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .title-block {
        background: linear-gradient(135deg, #FF3008, #FF6B35);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .title-block h1 { color: white; font-size: 2.5rem; margin: 0; }
    .title-block p { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2d3147;
    }
    .metric-card .label { color: #8b8fa8; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .metric-card .value { color: white; font-size: 1.5rem; font-weight: 700; }
    .result-box {
        background: linear-gradient(135deg, #1a1f36, #252b45);
        border: 2px solid #FF3008;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-box .duration { color: #FF3008; font-size: 3rem; font-weight: 800; }
    .result-box .subtitle { color: #8b8fa8; font-size: 1rem; }
    .breakdown-item {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        border-left: 4px solid #FF3008;
    }
    .section-title { color: #FF3008; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem; }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { color: #c0c4d6 !important; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="title-block">
    <h1>🍕 DoorDash Delivery Duration Predictor</h1>
    <p>ML-powered prediction using Ridge Regression + Artificial Neural Network</p>
</div>
""", unsafe_allow_html=True)

# --- Layout ---
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<p class="section-title">📦 Order Information</p>', unsafe_allow_html=True)
    market_id = st.selectbox("Market ID", [1, 2, 3, 4, 5, 6])
    store_category = st.selectbox("Store Category", sorted([
        "american", "mexican", "pizza", "chinese", "japanese", "italian",
        "indian", "thai", "burger", "sandwich", "fast", "dessert",
        "vietnamese", "mediterranean", "breakfast", "other"
    ]))
    order_protocol = st.selectbox("Order Protocol", [1, 2, 3, 4, 5, 6, 7])
    total_items = st.slider("Total Items", 1, 20, 3)
    subtotal = st.number_input("Subtotal (cents)", min_value=0, max_value=30000, value=2500, step=100)

    st.markdown('<p class="section-title" style="margin-top:1.5rem">🛵 Dasher Status</p>', unsafe_allow_html=True)
    total_onshift_dashers = st.slider("Total Onshift Dashers", 0, 150, 30)
    total_busy_dashers = st.slider("Total Busy Dashers", 0, 150, 20)
    total_outstanding_orders = st.slider("Outstanding Orders", 0, 200, 40)

    st.markdown('<p class="section-title" style="margin-top:1.5rem">⏱ Estimates</p>', unsafe_allow_html=True)
    estimated_order_place_duration = st.slider("Order Place Duration (sec)", 0, 600, 300)
    estimated_store_to_consumer_driving_duration = st.slider("Driving Duration (sec)", 0, 2000, 600)

    st.markdown('<p class="section-title" style="margin-top:1.5rem">🕐 Timing</p>', unsafe_allow_html=True)
    order_hour = st.slider("Order Hour", 0, 23, 19)
    order_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

with right_col:
    # --- Feature Engineering ---
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    order_day_of_week = day_map[order_day]
    busy_dasher_ratio = total_busy_dashers / total_onshift_dashers if total_onshift_dashers > 0 else 0
    orders_per_dasher = total_outstanding_orders / total_onshift_dashers if total_onshift_dashers > 0 else 0
    free_dashers = total_onshift_dashers - total_busy_dashers
    avg_item_price = subtotal / total_items if total_items > 0 else 0
    is_weekend = 1 if order_day_of_week >= 5 else 0
    is_rush_hour = 1 if (11 <= order_hour <= 13) or (17 <= order_hour <= 20) else 0
    day_cos = np.cos(2 * np.pi * order_day_of_week / 7)
    non_prep_duration = estimated_order_place_duration + estimated_store_to_consumer_driving_duration

    st.markdown('<p class="section-title">📊 Computed Signals</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Busy Dasher Ratio</div><div class="value">{busy_dasher_ratio:.2f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="label">Free Dashers</div><div class="value">{free_dashers}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="label">Rush Hour</div><div class="value">{"🔴 Yes" if is_rush_hour else "🟢 No"}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Orders per Dasher</div><div class="value">{orders_per_dasher:.2f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="label">Avg Item Price (¢)</div><div class="value">{avg_item_price:.0f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="label">Weekend</div><div class="value">{"🔴 Yes" if is_weekend else "🟢 No"}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    @st.cache_resource
    def load_models():
        ridge_model = joblib.load("ridge_model.pkl")
        ridge_scaler = joblib.load("ridge_scaler.pkl")
        feature_set = joblib.load("feature_set.pkl")
        ann_model = keras.models.load_model("ann_model.keras")
        ann_y_scaler = joblib.load("ann_y_scaler.pkl")
        return ridge_model, ridge_scaler, feature_set, ann_model, ann_y_scaler

    if st.button("🔮 Predict Delivery Duration", use_container_width=True, type="primary"):
        try:
            ridge_model, ridge_scaler, feature_set, ann_model, ann_y_scaler = load_models()

            input_data = {col: 0 for col in feature_set}
            input_data.update({
                "total_outstanding_orders": total_outstanding_orders,
                "avg_item_price": avg_item_price,
                "orders_per_dasher": orders_per_dasher,
                "subtotal": subtotal,
                "busy_dasher_ratio": busy_dasher_ratio,
                "free_dashers": free_dashers,
                "is_rush_hour": is_rush_hour,
                "is_weekend": is_weekend,
                "day_cos": day_cos,
                "non_prep_duration": non_prep_duration,
            })

            X_input = pd.DataFrame([input_data])[feature_set]
            X_scaled = ridge_scaler.transform(X_input)
            prep_time_pred = ridge_model.predict(X_scaled)[0]

            ann_input_raw = np.array([[prep_time_pred,
                                       estimated_store_to_consumer_driving_duration,
                                       estimated_order_place_duration]])
            ann_input_scaled = (ann_input_raw - ann_y_scaler.mean_[0]) / ann_y_scaler.scale_[0]
            y_pred_scaled = ann_model.predict(ann_input_scaled, verbose=0)
            total_duration = ann_y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
            minutes = total_duration / 60

            st.markdown(f"""
            <div class="result-box">
                <div class="subtitle">Estimated Delivery Duration</div>
                <div class="duration">{minutes:.0f} min</div>
                <div class="subtitle">{total_duration:.0f} seconds</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">🔍 Breakdown</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="breakdown-item"><span style="color:#c0c4d6">🧑‍🍳 Store Prep Time</span><span style="color:white;font-weight:600">{prep_time_pred:.0f} sec</span></div>
            <div class="breakdown-item"><span style="color:#c0c4d6">🚗 Driving Duration</span><span style="color:white;font-weight:600">{estimated_store_to_consumer_driving_duration} sec</span></div>
            <div class="breakdown-item"><span style="color:#c0c4d6">📋 Order Placement</span><span style="color:white;font-weight:600">{estimated_order_place_duration} sec</span></div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with Streamlit · Ridge Regression + ANN Pipeline · DoorDash Delivery Duration Prediction")
