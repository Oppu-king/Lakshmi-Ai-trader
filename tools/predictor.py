# tools/predictor.py
import random

def predict_price_trend(symbol):
    outcomes = ["Uptrend likely 📈", "Downtrend expected 📉", "Sideways movement 🔁"]
    return f"Prediction for {symbol}: {random.choice(outcomes)}"
