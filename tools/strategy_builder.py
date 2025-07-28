# tools/strategy_builder.py
def build_strategy(indicator, threshold):
    if indicator == "rsi":
        if threshold < 30:
            return "Buy signal: RSI below 30 indicates oversold."
        elif threshold > 70:
            return "Sell signal: RSI above 70 indicates overbought."
        else:
            return "Hold: RSI within neutral range."
    elif indicator == "macd":
        return "Strategy based on MACD crossovers."
    else:
        return "No strategy available for this indicator.
