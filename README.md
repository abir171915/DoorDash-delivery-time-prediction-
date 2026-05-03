# 🍕 How Long Will My DoorDash Take?

> *"Your order has been placed. Estimated delivery: 35 minutes."*
> 
> But how does DoorDash know that? And can we do better?

That's exactly what this project set out to answer.

---

## The Problem

Every time you order food on DoorDash, the app shows you an estimated delivery time. That number has a huge impact as it sets your expectations and determines how satisfied you feel when the food arrives.

But predicting delivery time is harder than it sounds because it depends on:
- How busy the dashers are right now
- What kind of food you ordered
- What time of day it is
- How far the restaurant is
- How long the store takes to prepare the food
- And to be honest so many factors

The last one, **store prep time**  is the hardest to predict. And that's where this project focuses.

---

## The Dataset

We worked with ~197,000 real DoorDash orders, each containing information available at the moment the order was placed:

- How many dashers are on shift and how many are already busy
- Order details (items, price, category)
- Time of order
- Estimated driving and placement durations

The goal: **predict how long the full delivery will take, in seconds.**

---

## The Journey

### Step 1 — Understanding the Data

Before building any model, we spent time really understanding what we are working with. Missing values, outliers, weird distributions all of it needed to be addressed.

One thing that stood out immediately: the maximum delivery duration was **~98 days**. Clearly bad data. These outliers were carefully handled before any modelling began.

### Step 2 — Creating Meaningful Features

Raw data rarely tells the full story. We engineered features that better capture the real signals:

- **Dasher pressure** : if 95% of dashers are busy, deliveries will take longer. So we created `busy_dasher_ratio` and `orders_per_dasher`
- **Time of day** : rush hour (lunch and dinner) means more orders and longer waits. We flagged these periods and also used cyclical encoding (sin/cos) so the model understands that 11pm and midnight are close in time
- **Order complexity** : average item price, price range and subtotal as signals for order size and complexity

### Step 3 — Cleaning Up Redundant Features

More features isn't always better. We checked for features that were essentially saying the same thing twice using:

- **Correlation analysis** — removed pairs with correlation > 0.8
- **VIF (Variance Inflation Factor)** — removed features causing multicollinearity
- **Gini Importance** — let a Random Forest tell us which features actually matter for prediction

This narrowed us down to the **top 40 most informative features**.

### Step 4 — A Smarter Way to Frame the Problem

Here's where things got interesting.

Instead of predicting the total delivery duration directly, we realised something: DoorDash already has good estimates for driving time and order placement time. The only truly unknown piece is **how long the store takes to prepare the food**.

So we split the problem:

```
Total Duration = Store Prep Time + Driving Time + Order Placement Time
                      ↑
               This is what we predict
```

This is like predicting one ingredient of a recipe instead of guessing the whole dish. Much easier and more accurate.

### Step 5 — Building the Models

We trained a **Ridge Regression** model to predict store prep time, then fed that prediction (along with the known driving and placement estimates) into an **XGBoost model** to get the final delivery duration.

We tested 6 different models along the way:

| Model | RMSE |
|---|---|
| **DecisionTree** | **1,044 sec (~17 min)** |
| MLP Neural Network | 1,048 sec |
| LGBM | 1,053 sec |
| Linear / Ridge | 1,054 sec |
| XGBoost | 1,065 sec |
| Random Forest | 1,215 sec |

### Step 6 — Bringing It to Life

The final model was deployed as an interactive web app where you can input order details and get a predicted delivery time in real time.

---

## Try It Yourself

**Live demo:** [https://sx8mnutq4zmulkozjqffct.streamlit.app/]

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## What's Inside

```
├── fulleda.ipynb       # The full story — EDA, features, models
├── app.py              # Streamlit web application
├── requirements.txt    # Dependencies
├── ridge_model.pkl     # Stage 1: prep time predictor
├── final_model.pkl     # Stage 2: total duration predictor
└── feature_set.pkl     # Selected features
```

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `statsmodels` · `Streamlit`

---

*Built as an end-to-end machine learning project — from raw data exploration to deployed web application.*
