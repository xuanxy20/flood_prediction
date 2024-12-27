import streamlit as st
import joblib
import numpy as np

from constants import MODEL_PATH, SCALER_PATH

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Streamlit App
st.title("Flood Probability Prediction")
st.markdown("""
This app predicts the probability of flooding based on various environmental, 
infrastructural, and planning-related factors. Adjust the inputs below to get a prediction.
""")

fields = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices",
    "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability",
    "Landslides", "Watersheds", "DeterioratingInfrastructure", "PopulationScore",
    "WetlandLoss", "InadequatePlanning", "PoliticalFactors"
]

# Determine the number of columns
num_columns = 3
cols = st.columns(num_columns)

# List to store the input values
input_data = []

# Distribute the fields across the columns and collect input
for idx, field in enumerate(fields):
    col = cols[idx % num_columns]
    value = col.number_input(
        field,
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )
    input_data.append(value)

# Predict button
if st.button("Predict Flood Probability"):
    with st.spinner("Calculating flood probability..."):
        # Scale the input data
        new_data = np.array(input_data).reshape(1, -1)
        new_data_scaled = scaler.transform(new_data)

        # Predict flood probability
        flood_probability = model.predict(new_data_scaled)
        st.success(f"Predicted Flood Probability: {flood_probability[0]:.2f}")
