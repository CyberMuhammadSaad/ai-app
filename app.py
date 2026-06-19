import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Property Pro", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
# 🏠 AI Property Intelligence Platform
### Professional ML Dashboard
""")

st.divider()

# ---------------- DATA ----------------
df = pd.DataFrame({
    "size": [300, 500, 700, 900, 1100, 1300, 1500, 1700, 2000, 2300, 2600, 3000],
    "type": ["Apartment", "Apartment", "House", "House", "House",
             "Villa", "Villa", "Villa", "Commercial", "Commercial", "Commercial", "Villa"],
    "market": ["Low", "Low", "Normal", "Normal", "Normal",
               "High", "High", "High", "High", "High", "High", "High"],
    "price": [12, 18, 25, 35, 45, 60, 75, 90, 120, 150, 175, 200]
})

X = df[["size", "type", "market"]]
y = df["price"]

# ---------------- MODEL ----------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["type", "market"])
], remainder="passthrough")

model = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

pred_test = pipeline.predict(X_test)
r2 = r2_score(y_test, pred_test)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Controls")

size = st.sidebar.slider("Property Size (sq ft)", 100, 3000, 1200)
ptype = st.sidebar.selectbox("Property Type", df["type"].unique())
market = st.sidebar.selectbox("Market Condition", df["market"].unique())

predict_btn = st.sidebar.button("🚀 Predict Price")

# ---------------- TOP METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", len(df))
col2.metric("Model Accuracy (R²)", f"{r2:.2f}")
col3.metric("Model Type", "Random Forest")

st.divider()

# ---------------- PREDICTION ----------------
if predict_btn:

    input_df = pd.DataFrame([{
        "size": size,
        "type": ptype,
        "market": market
    }])

    prediction = pipeline.predict(input_df)[0]

    st.subheader("📊 Prediction Result")

    c1, c2 = st.columns(2)

    c1.markdown(f"""
    ### 🏷️ Estimated Price  
    ## {prediction:.2f} Lakh PKR
    """)

    c2.markdown(f"""
    ### 📌 Input Summary  
    - Size: {size} sq ft  
    - Type: {ptype}  
    - Market: {market}  
    """)

    st.divider()

    # ---------------- GRAPH ----------------
    st.subheader("📈 Market Trend Visualization")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["size"],
        y=df["price"],
        mode="markers+lines",
        name="Market Data"
    ))

    fig.add_trace(go.Scatter(
        x=[size],
        y=[prediction],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Your Property"
    ))

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER INSIGHT ----------------
st.divider()

st.subheader("🧠 AI Insight")

st.info("""
Prices increase mainly with size and market demand.  
Luxury properties (Villa/High market) significantly increase valuation.
""")
