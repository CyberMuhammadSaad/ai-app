import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Property Engine Pro", layout="wide")

# ---------------- CLEAN UI STYLE ----------------
st.markdown("""
    <style>
        .main {
            background-color: #0f172a;
            color: white;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 10px;
            height: 45px;
            width: 100%;
            font-size: 16px;
        }
        .stMetric {
            background-color: #1e293b;
            padding: 15px;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🏠 AI Property Price Intelligence System")
st.caption("Advanced ML-powered valuation dashboard")

st.divider()

# ---------------- MODEL ----------------
X = np.array([[500], [800], [1000], [1200], [1500], [1800]])
y = np.array([15, 28, 40, 55, 70, 90])

model = LinearRegression()
model.fit(X, y)

def market_factor(region):
    return {
        "Low Demand": 0.85,
        "Normal": 1.0,
        "High Demand": 1.25
    }[region]

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Property Configuration")

size = st.sidebar.slider("House Size (sq ft)", 100, 3000, 1000, step=50)
region = st.sidebar.selectbox("Market Type", ["Low Demand", "Normal", "High Demand"])
ptype = st.sidebar.selectbox("Property Type", ["Apartment", "House", "Villa", "Commercial"])

predict_btn = st.sidebar.button("Run AI Prediction")

# ---------------- MAIN LAYOUT ----------------
col1, col2, col3 = st.columns(3)

if predict_btn:

    base = model.predict([[size]])[0]
    final = base * market_factor(region)

    # ---------------- KPI CARDS ----------------
    col1.metric("Base Price", f"{base:.2f} Lakh")
    col2.metric("Final Price", f"{final:.2f} Lakh")
    col3.metric("Market Type", region)

    st.divider()

    # ---------------- BREAKDOWN ----------------
    st.subheader("📊 Prediction Breakdown")

    st.write(f"• Property Type: **{ptype}**")
    st.write(f"• Size Input: **{size} sq ft**")
    st.write(f"• Market Condition: **{region}**")
    st.write(f"• Adjustment Factor: **{market_factor(region)}x**")

    st.divider()

    # ---------------- BETTER GRAPH ----------------
    st.subheader("📈 Market Intelligence Graph")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(X, y, marker="o", linewidth=3, label="Market Trend", color="#60a5fa")
    ax.scatter(size, final, color="red", s=150, label="Your Property")

    ax.set_title("Property Price vs Size", fontsize=14)
    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Price (Lakh PKR)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    # ---------------- INSIGHT BOX ----------------
    st.info("This model uses linear regression on sample data. For real accuracy, use real estate datasets.")
