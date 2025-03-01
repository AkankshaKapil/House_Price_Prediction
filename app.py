import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
with open("linear_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scalers.pkl", "rb") as scaler_file:
    scaler_x, scaler_y = pickle.load(scaler_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data and convert to correct types
        features = [
            float(request.form["area"]),
            int(request.form["bedrooms"]),
            int(request.form["bathrooms"]),
            int(request.form["stories"]),
            int(request.form["mainroad"]),
            int(request.form["guestroom"]),
            int(request.form["basement"]),
            int(request.form["hotwaterheating"]),
            int(request.form["airconditioning"]),
            int(request.form["parking"]),
            int(request.form["prefarea"]),
            int(request.form["furnishingstatus"]),
        ]
        
        # Convert to NumPy array and reshape for model
        features_array = np.array([features])

        # Scale input data
        features_scaled = scaler_x.transform(features_array)

        # Make prediction
        price_scaled = model.predict(features_scaled)

        # Inverse transform to get actual price
        price = scaler_y.inverse_transform(price_scaled.reshape(-1, 1))[0][0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ₹{price:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
