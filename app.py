import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Property Price Predictor", layout="wide")

# ---------------- TITLE ----------------
st.title("🏠 AI Property Price Prediction System")
st.caption("Simple ML model using Linear Regression")

st.divider()

# ---------------- DATA (ML MODEL) ----------------
# Training data (size in sq ft → price in Lakh PKR)
X = np.array([[500], [800], [1000], [1200], [1500], [1800]])
y = np.array([15, 28, 40, 55, 70, 90])

model = LinearRegression()
model.fit(X, y)

# ---------------- INPUT ----------------
size = st.number_input("Enter House Size (sq ft)", min_value=100, step=50)

region = st.selectbox("Select Market Type", ["Low Demand", "Normal", "High Demand"])

property_type = st.selectbox("Property Type", ["Apartment", "House", "Villa", "Commercial"])

# market adjustment
def market_factor(region):
    if region == "Low Demand":
        return 0.85
    elif region == "Normal":
        return 1.0
    else:
        return 1.25

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):

    base_price = model.predict([[size]])[0]
    final_price = base_price * market_factor(region)

    st.success("Prediction Completed")

    # RESULT
    st.markdown("## 💰 Estimated Price")
    st.markdown(f"### {final_price:.2f} Lakh PKR")

    st.write("### Breakdown")
    st.write(f"- Property Type: {property_type}")
    st.write(f"- Size: {size} sq ft")
    st.write(f"- Market: {region}")
    st.write(f"- Base Price: {base_price:.2f} Lakh PKR")

    # ---------------- GRAPH ----------------
    st.markdown("## 📊 Model Visualization")

    fig, ax = plt.subplots()
    ax.plot(X, y, marker="o", label="Training Data")
    ax.scatter(size, final_price, color="red", label="Your Property")

    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Price (Lakh PKR)")
    ax.legend()

    st.pyplot(fig)
