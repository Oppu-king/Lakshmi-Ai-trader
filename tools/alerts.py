# tools/alerts.py
def generate_alert(symbol, condition):
    if condition == "breakout":
        return f"🚨 Alert: {symbol} is breaking out of resistance!"
    elif condition == "breakdown":
        return f"⚠️ Alert: {symbol} is breaking below support!"
    else:
        return f"No alert condition met for {symbol}."
