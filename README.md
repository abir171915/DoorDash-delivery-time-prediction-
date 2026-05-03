# DoorDash Delivery Duration Prediction

A machine learning project that predicts how long a DoorDash delivery will take, built end-to-end from raw data to a deployed web application.

## Problem Statement

DoorDash needs to accurately estimate delivery durations to improve customer experience and dasher efficiency. Given information available at the time an order is placed (number of dashers, order size, time of day, store category, etc.), can we predict how long the delivery will take?

## Dataset

- **Source:** DoorDash historical delivery data
- **Size:** ~197,000 orders
- **Target variable:** `delivery_duration` (seconds from order placement to delivery)

## Project Structure

```
├── fulleda.ipynb       # Full EDA, feature engineering, and model training
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── ridge_model.pkl     # Trained Ridge regression model (prep time predictor)
├── ridge_scaler.pkl    # StandardScaler for Ridge model features
├── feature_set.pkl     # Selected features list
└── final_model.pkl     # XGBoost final model (total duration predictor)
```

## Approach

The problem was broken into two stages rather than predicting total delivery duration directly:

```
Total Delivery Duration = Prep Time + Driving Time + Order Placement Time
```

Since driving time and order placement time are already estimated by DoorDash, the only unknown is **store prep time**. So:

1. **Stage 1 — Ridge Regression:** Predicts store prep time from order features
2. **Stage 2 — XGBoost:** Takes predicted prep time + known estimates → predicts final delivery duration

## Feature Engineering

| Feature | Description |
|---|---|
| `busy_dasher_ratio` | total_busy / total_onshift dashers |
| `orders_per_dasher` | Outstanding orders per available dasher |
| `free_dashers` | Absolute idle dasher count |
| `avg_item_price` | subtotal / total_items |
| `hour_sin / hour_cos` | Cyclical encoding of order hour |
| `day_sin / day_cos` | Cyclical encoding of day of week |
| `is_rush_hour` | 1 if lunch (11–1pm) or dinner (5–8pm) |
| `is_weekend` | 1 if Saturday or Sunday |
| `non_prep_duration` | Sum of estimated driving + order placement |

One-hot encoding was applied to: `market_id`, `order_protocol`, `store_primary_category`

## Feature Selection

Three methods were used to select the most important features:

1. **Correlation analysis** — removed highly correlated feature pairs (threshold > 0.8)
2. **VIF (Variance Inflation Factor)** — removed features with VIF > 20 to reduce multicollinearity
3. **Gini Importance (Random Forest)** — selected top 40 most predictive features

## Model Results

All models trained on the final 3-feature input (predicted prep time + 2 known estimates):

| Model | RMSE (seconds) | RMSE (minutes) |
|---|---|---|
| **DecisionTree** | **1,044** | **~17.4 min** |
| MLP | 1,048 | ~17.5 min |
| LGBM | 1,053 | ~17.6 min |
| LinearReg / Ridge | 1,054 | ~17.6 min |
| XGBoost | 1,065 | ~17.8 min |
| RandomForest | 1,215 | ~20.3 min |

## Web Application

An interactive Streamlit dashboard was built where users can input order details and get a real-time delivery duration prediction.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Models** — Ridge Regression, XGBoost, LightGBM, Random Forest, ANN
- **Feature selection** — VIF (statsmodels), Gini importance (sklearn)
- **Web app** — Streamlit
- **Deployment** — Streamlit Cloud
