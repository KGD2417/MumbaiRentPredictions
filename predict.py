import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("rent_price_model.pkl")
location_encoder = joblib.load("location_encoder.pkl")
type_encoder = joblib.load("type_encoder.pkl")
furnishing_encoder = joblib.load("furnishing_encoder.pkl")

def predict_rent(locality, house_type, build_up_area, carpet_area, furnishing, bathrooms, balcony, parking):
    # Encode categorical inputs
    locality_encoded = location_encoder.transform([locality])[0]
    type_encoded = type_encoder.transform([house_type])[0]
    furnishing_encoded = furnishing_encoder.transform([furnishing])[0]

    # Prepare input data
    input_data = np.array([[locality_encoded, type_encoded, build_up_area, carpet_area, furnishing_encoded, bathrooms, balcony, parking]])

    # Predict rent
    predicted_rent = model.predict(input_data)[0]
    return round(predicted_rent, 2)
