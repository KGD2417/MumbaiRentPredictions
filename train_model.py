import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = "Mumbai_House_Rent.csv"
data = pd.read_csv(file_path)

# Data Cleaning
data = data.replace("Missing", np.nan)
data["Balcony"] = pd.to_numeric(data["Balcony"], errors="coerce")
data["Carpet_area(sq.ft)"] = data["Carpet_area(sq.ft)"].str.replace(" sq.ft", "", regex=True).astype(float)
data["Build_up_area(sq.ft)"] = data["Build_up_area(sq.ft)"].str.replace(" sq.ft", "", regex=True).astype(float)

# Filling missing values
data.fillna(data.median(numeric_only=True), inplace=True)

# Encoding categorical variables
le = LabelEncoder()
data["Locality"] = le.fit_transform(data["Locality"])
data["Type"] = le.fit_transform(data["Type"])
data["Furnishing"] = le.fit_transform(data["Furnishing"])

# Selecting Features & Target
X = data.drop(columns=["Rent/Month"])
y = data["Rent/Month"]

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "rent_price_model.pkl")

print("Model training complete! Saved as rent_price_model.pkl")
