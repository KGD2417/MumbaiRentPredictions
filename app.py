import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("rent_price_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
type_encoder = joblib.load("type_encoder.pkl")
furnishing_encoder = joblib.load("furnishing_encoder.pkl")

# Streamlit UI
st.title("🏠 Mumbai House Rent Prediction")

# User Inputs
locality = st.selectbox("Locality", location_encoder.classes_)
house_type = st.selectbox("House Type", type_encoder.classes_)
build_up_area = st.number_input("Build Up Area (sq.ft)", min_value=100)
carpet_area = st.number_input("Carpet Area (sq.ft)", min_value=100)
furnishing = st.selectbox("Furnishing", furnishing_encoder.classes_)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, step=1)
balcony = st.number_input("Balcony", min_value=0, max_value=5, step=1)
parking = st.number_input("Parking", min_value=0, max_value=3, step=1)

# Predict Rent Button
if st.button("Predict Rent"):
    # Encode categorical inputs
    locality_encoded = location_encoder.transform([locality])[0]
    type_encoded = type_encoder.transform([house_type])[0]
    furnishing_encoded = furnishing_encoder.transform([furnishing])[0]

    # Prepare input data
    input_data = np.array([[locality_encoded, type_encoded, build_up_area, carpet_area, furnishing_encoded, bathrooms, balcony, parking]])

    # Predict rent
    predicted_rent = model.predict(input_data)[0]
    st.success(f"💰 Estimated Rent: ₹{round(predicted_rent, 2)}")
