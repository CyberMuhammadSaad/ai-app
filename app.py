import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("My First AI App")

sizes = np.array([[500], [800], [1000], [1200], [1500]])
prices = np.array([50, 80, 100, 130, 160])

model = LinearRegression()
model.fit(sizes, prices)

size = st.number_input("Enter house size")

if st.button("Predict"):
    result = model.predict([[size]])
    st.write("Predicted price:", result[0])
