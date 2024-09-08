import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Set up the Streamlit app title and description
st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of having diabetes based on user input medical parameters.")

# Create input fields for user input
pregnancies = st.slider('Number of Pregnancies', 0, 15, 1)
glucose = st.slider('Glucose Level', 0, 200, 100)
blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 150, 70)
skin_thickness = st.slider('Skin Thickness (mm)', 0, 100, 20)
insulin = st.slider('Insulin Level (mu U/ml)', 0, 900, 80)
bmi = st.slider('Body Mass Index (BMI)', 0.0, 70.0, 21.0)
diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
age = st.slider('Age', 21, 100, 30)

# Organize inputs into a dataframe with correct feature names
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                          columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])  # Use the original feature names

# Predict using the model
prediction = model.predict(input_data)

# Get the predicted label directly
predicted_outcome = prediction[0]  # Use the string prediction directly

# Display the prediction result
st.write(f"The predicted outcome is: **{predicted_outcome}**")
