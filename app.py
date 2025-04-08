from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model and label encoder as None
model = None
label_encoder = None

def load_models():
    global model, label_encoder
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "layoff_risk_model.pkl")
        encoder_path = os.path.join(current_dir, "label_encoder.pkl")
        
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Encoder path: {encoder_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")
            
        logger.info("Loading model and label encoder...")
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        logger.info("Model and label encoder loaded successfully")
        
        # Log model type and attributes
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model attributes: {dir(model)}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(f"Python version: {sys.version}")
        logger.error(f"Joblib version: {joblib.__version__}")
        raise

# Load models when the app starts
try:
    load_models()
except Exception as e:
    logger.error(f"Failed to load models during startup: {str(e)}")

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
    if model is None or label_encoder is None:
        try:
            load_models()
        except Exception as e:
            return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    data = request.get_json()
    logger.info(f"Received prediction request with data: {data}")

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

        logger.info(f"Prediction successful: {prediction_label}")
        return jsonify({"layoff_risk": prediction_label})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Updated port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)