
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction App")

# Input fields using sliders
age = st.slider("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.slider("Chest Pain Type (cp)", 0, 3, 0)
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Serum Cholestoral in mg/dl (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.slider("Resting Electrocardiographic Results (restecg)", 0, 2, 1)
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.slider("Slope of the Peak Exercise ST Segment (slope)", 0, 2, 1)
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (ca)", 0, 4, 0)
thal = st.slider("Thalassemia (thal)", 0, 3, 2)

# Predict
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    st.success(result)




# input_df = user_input_features()

# # Preprocess input
# input_scaled = scaler.transform(input_df)

# # Prediction
# prediction = model.predict(input_scaled)
# # prediction_proba = model.predict_proba(input_scaled)

# # Output
# st.subheader("Prediction Result")
# st.write("🧬 Diabetic" if prediction[0] == 1 else "✅ Not Diabetic")
