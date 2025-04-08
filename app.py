from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("layoff_risk_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define expected input features (same as during training)
categorical_features = [
    "company_name", "company_location", "reporting_quarter", "economic_condition_tag",
    "past_layoffs", "job_title", "department", "remote_work", "industry"
]

numerical_features = [
    "revenue_growth", "profit_margin", "stock_price_change", "total_employees",
    "years_at_company", "salary_range", "performance_rating",
    "industry_layoff_rate", "unemployment_rate", "inflation_rate"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Combine inputs into a DataFrame
        input_df = pd.DataFrame([data])

        # Add engineered features (must match training code)
        input_df["employee_stability"] = input_df["years_at_company"] * input_df["performance_rating"]
        input_df["economic_pressure"] = input_df["inflation_rate"] + input_df["unemployment_rate"] - input_df["revenue_growth"]

        # Ensure the columns are in the correct order
        all_features = categorical_features + numerical_features + ["employee_stability", "economic_pressure"]
        input_df = input_df[all_features]

        # Predict
        prediction_encoded = model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        return jsonify({"layoff_risk": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)