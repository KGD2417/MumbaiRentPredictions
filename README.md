# Mumbai House Rent Prediction AI

## 📌 Overview
This project is a machine learning model that predicts **house rent prices in Mumbai** based on various factors like locality, BHK type, furnishing, carpet area, and more. The model is trained using **Decision Tree Regression** and is deployed with a **Streamlit UI** for easy interaction.

## 🚀 Features
- **Predict house rent** based on user inputs.
- **Uses Decision Tree Regressor** for accurate price estimation.
- **Streamlit Web UI** for easy interaction.
- **FastAPI Backend (Optional)** to expose an API for predictions.

## 📂 Dataset
The dataset used for training is available on [Kaggle](https://www.kaggle.com/datasets/omkargangan/mumbai-house-rent). It contains the following features:

- `Locality` (e.g., Andheri, Bandra, Dadar)
- `Type` (e.g., 1 BHK, 2 BHK, etc.)
- `Rent/Month`
- `Build_up_area (sq.ft)`
- `Furnishing` (Unfurnished, Semi-furnished, Fully-furnished)
- `Bathrooms`
- `Balcony`
- `Parking`
- `Carpet_area (sq.ft)`

## 🛠 Installation & Setup
### 1️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed. Then, run:
```bash
pip install pandas numpy scikit-learn joblib streamlit fastapi uvicorn
```

### 2️⃣ Train the Model
Run the script to train and save the model:
```bash
python train_model.py
```
This will generate a `rent_price_model.pkl` file.

### 3️⃣ Run the Streamlit UI
```bash
streamlit run app.py
```
Access the UI at **http://localhost:8501**.

### 4️⃣ (Optional) Run the FastAPI Backend
If you want an API instead of UI, run:
```bash
uvicorn api:app --reload
```
Test API at: **http://127.0.0.1:8000/predict?locality=Andheri&type=2%20BHK&area=1200**

## 🖥️ File Structure
```
📂 MumbaiRentPredictions
│-- 📄 train_model.py  # Train and save the model
│-- 📄 app.py          # Streamlit UI for predictions
│-- 📄 api.py          # FastAPI backend for predictions
│-- 📄 rent_price_model.pkl  # Saved ML model
│-- 📄 requirements.txt  # List of dependencies
```

## 🏆 Future Improvements
- Improve model accuracy with **advanced ML algorithms**.
- Deploy on **AWS/GCP**.
- Add **more features like location-based predictions**.

## 🤝 Contributing
Feel free to **fork this repo** and contribute by improving the model or UI!

## 📜 License
This project is **MIT licensed**. Feel free to use and modify it.

