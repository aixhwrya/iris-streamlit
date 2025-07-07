import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model
model = joblib.load('iris_model.pkl')
iris = load_iris()

st.title("🌼 Iris Flower Classifier")
st.write("Move the sliders to input flower features:")

# Inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    st.success(f"Predicted Species: **{iris.target_names[prediction].title()}**")

    # Visualization
    fig, ax = plt.subplots()
    ax.pie(probabilities, labels=iris.target_names, autopct='%1.1f%%')
    st.pyplot(fig)
