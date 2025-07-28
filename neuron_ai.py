# neuron_ai.py
import random
import json
import os
from datetime import datetime

NEURON_FILE = "neuron_memory.json"

# Load or initialize neuron memory
def load_memory():
    if os.path.exists(NEURON_FILE):
        with open(NEURON_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "history": [],
            "patterns": {},
            "last_signal": None
        }

def save_memory(memory):
    with open(NEURON_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def analyze_pattern(history):
    if len(history) < 5:
        return "❗Not enough data", 50

    last = history[-1]
    avg = sum(history[-5:]) / 5
    trend = "UP" if last > avg else "DOWN"

    score = abs(last - avg) / avg * 100
    confidence = round(min(score * 2, 99), 2)

    if trend == "UP":
        return "📈 Bullish Surge Detected", confidence
    else:
        return "📉 Bearish Collapse Detected", confidence

def ingest_price(price):
    memory = load_memory()
    memory["history"].append(price)

    signal, confidence = analyze_pattern(memory["history"])

    memory["last_signal"] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price": price,
        "signal": signal,
        "confidence": confidence
    }

    save_memory(memory)

    return {
        "signal": signal,
        "confidence": confidence,
        "latest_price": price,
        "memory_size": len(memory["history"]),
        "time": memory["last_signal"]["time"]
    
