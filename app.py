import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Config
st.set_page_config(page_title="METABRIC Survival Predictor", layout="centered")

# 2. Load the saved model
try:
    model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'breast_cancer_model.pkl' is in the same folder.")

st.title("🎀 METABRIC Survival Predictor")
st.markdown("---")

# 3. Sidebar for Patient Inputs
st.sidebar.header("Patient Clinical Data")

# Adjust these inputs based on the features you used to train your model
age = st.sidebar.slider("Age at Diagnosis", 20, 95, 50)
tumor_size = st.sidebar.number_input("Tumor Size (mm)", 1.0, 200.0, 25.0)
tumor_stage = st.sidebar.selectbox("Tumor Stage", [1, 2, 3, 4])

# 4. Main Prediction Logic
if st.button("Calculate Survival Probability"):
    # Prepare the input array (Order MUST match your X_train columns)
    # Example: [Age, Tumor_Size, Tumor_Stage]
    input_data = np.array([[age, tumor_size, tumor_stage]])
    
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Results")
        if prediction[0] == 1 or prediction[0] == 'Living': # Adjust based on your label encoding
            st.success(f"**Outcome:** Likely to Survive")
        else:
            st.warning(f"**Outcome:** High Risk / Deceased")
            
        st.metric(label="Survival Probability", value=f"{probability:.2%}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Developed for MBA Data Science Project - METABRIC Analysis")