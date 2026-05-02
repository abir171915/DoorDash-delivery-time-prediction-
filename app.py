import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras

st.set_page_config(page_title="DoorDash Delivery Duration Predictor", page_icon="🚗", layout="centered")

st.title("🚗 DoorDash Delivery Duration Predictor")
st.markdown("Predict how long a delivery will take based on order and market details.")

st.sidebar.header("Order Details")

# --- Sidebar Inputs ---
market_id = st.sidebar.selectbox("Market ID", [1, 2, 3, 4, 5, 6])
store_category = st.sidebar.selectbox("Store Category", [
    "american", "mexican", "pizza", "chinese", "japanese", "italian",
    "indian", "thai", "burger", "sandwich", "fast", "dessert",
    "vietnamese", "mediterranean", "breakfast", "other"
])
order_protocol = st.sidebar.selectbox("Order Protocol", [1, 2, 3, 4, 5, 6, 7])
total_items = st.sidebar.slider("Total Items", 1, 20, 3)
subtotal = st.sidebar.number_input("Subtotal (cents)", min_value=0, max_value=30000, value=2500, step=100)
total_onshift_dashers = st.sidebar.slider("Total Onshift Dashers", 0, 150, 30)
total_busy_dashers = st.sidebar.slider("Total Busy Dashers", 0, 150, 20)
total_outstanding_orders = st.sidebar.slider("Total Outstanding Orders", 0, 200, 40)
estimated_order_place_duration = st.sidebar.slider("Estimated Order Place Duration (sec)", 0, 600, 300)
estimated_store_to_consumer_driving_duration = st.sidebar.slider("Estimated Driving Duration (sec)", 0, 2000, 600)
order_hour = st.sidebar.slider("Order Hour", 0, 23, 19)
order_day = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

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

# --- Display computed features ---
st.subheader("Computed Features")
col1, col2, col3 = st.columns(3)
col1.metric("Busy Dasher Ratio", f"{busy_dasher_ratio:.2f}")
col2.metric("Orders per Dasher", f"{orders_per_dasher:.2f}")
col3.metric("Free Dashers", free_dashers)

col1, col2, col3 = st.columns(3)
col1.metric("Avg Item Price (¢)", f"{avg_item_price:.0f}")
col2.metric("Is Rush Hour", "Yes" if is_rush_hour else "No")
col3.metric("Is Weekend", "Yes" if is_weekend else "No")

# --- Load models (cached) ---
@st.cache_resource
def load_models():
    ridge_model = joblib.load("ridge_model.pkl")
    ridge_scaler = joblib.load("ridge_scaler.pkl")
    feature_set = joblib.load("feature_set.pkl")
    ann_model = keras.models.load_model("ann_model.keras")
    ann_X_scaler = joblib.load("ann_X_scaler.pkl")
    ann_y_scaler = joblib.load("ann_y_scaler.pkl")
    return ridge_model, ridge_scaler, feature_set, ann_model, ann_X_scaler, ann_y_scaler

# --- Predict Button ---
st.markdown("---")
if st.button("🔮 Predict Delivery Duration", use_container_width=True):
    try:
        ridge_model, ridge_scaler, feature_set, ann_model, ann_X_scaler, ann_y_scaler = load_models()

        # Build full feature dict — zeros for unused features, fill known ones
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

        # ANN prediction — scale each input value using y_scaler (same scaler was used for all)
        ann_input_raw = np.array([[prep_time_pred,
                                   estimated_store_to_consumer_driving_duration,
                                   estimated_order_place_duration]])
        ann_input_scaled = (ann_input_raw - ann_y_scaler.mean_[0]) / ann_y_scaler.scale_[0]
        y_pred_scaled = ann_model.predict(ann_input_scaled, verbose=0)
        total_duration = ann_y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

        minutes = total_duration / 60
        st.success(f"### Estimated Delivery Duration: **{total_duration:.0f} seconds ({minutes:.1f} minutes)**")

        st.markdown("**Breakdown:**")
        st.write(f"- Estimated Prep Time: `{prep_time_pred:.0f} sec`")
        st.write(f"- Estimated Driving: `{estimated_store_to_consumer_driving_duration} sec`")
        st.write(f"- Estimated Order Placement: `{estimated_order_place_duration} sec`")

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Make sure all .pkl and .keras files are in the same folder as app.py")

st.markdown("---")
st.caption("Built with Streamlit | DoorDash Delivery Duration Prediction Project")
