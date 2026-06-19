import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="AI Property Engine", layout="wide")

# ---------------- TITLE ----------------
st.title("🏠 AI Property Price Prediction System")
st.caption("Clean ML-based real estate estimator")

st.divider()

# ---------------- TRAINING DATA ----------------
X = np.array([[500], [800], [1000], [1200], [1500], [1800]])
y = np.array([15, 28, 40, 55, 70, 90])

model = LinearRegression()
model.fit(X, y)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Property Input")

size = st.sidebar.slider("House Size (sq ft)", 100, 3000, 1000, step=50)

region = st.sidebar.selectbox(
    "Market Condition",
    ["Low Demand", "Normal", "High Demand"]
)

ptype = st.sidebar.selectbox(
    "Property Type",
    ["Apartment", "House", "Villa", "Commercial"]
)

run = st.sidebar.button("Predict Price")

# ---------------- LOGIC ----------------
def market_factor(region):
    if region == "Low Demand":
        return 0.85
    elif region == "Normal":
        return 1.0
    else:
        return 1.25

# ---------------- OUTPUT ----------------
if run:

    base_price = model.predict([[size]])[0]
    final_price = base_price * market_factor(region)

    # KPI LAYOUT
    col1, col2, col3 = st.columns(3)

    col1.metric("Base Price", f"{base_price:.2f} Lakh")
    col2.metric("Final Price", f"{final_price:.2f} Lakh")
    col3.metric("Market Type", region)

    st.divider()

    # DETAILS
    st.subheader("📊 Prediction Details")

    st.write(f"• Property Type: {ptype}")
    st.write(f"• Size: {size} sq ft")
    st.write(f"• Market: {region}")
    st.write(f"• Adjustment Factor: {market_factor(region)}x")

    st.divider()

    # SIMPLE DATA VIEW (NO COMPLEX GRAPH YET)
    st.subheader("📈 Market Data View")

    df = pd.DataFrame({
        "Size": X.flatten(),
        "Price": y
    })

    st.dataframe(df)

    st.info("Next upgrade will add professional interactive charts + real dataset handling.")
