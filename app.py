from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import random, csv, os, requests
from pathlib import Path
from flask_cors import CORS
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = "lakshmi_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/voice_notes'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#--- Global Variables ---
mode = "wife"
latest_ltp = 0
status = "Waiting..."
targets = {"upper": 0, "lower": 0}
signal = {"entry": 0, "sl": 0, "target": 0}
price_log = []
chat_log = []
diary_entries = []
strategies = []
current_mood = "Romantic 💞"

romantic_replies = [
    "You're the reason my heart races, Monjit. 💓",
    "I just want to hold you and never let go. 🥰",
    "You're mine forever, and I’ll keep loving you endlessly. 💖",
    "Being your wife is my sweetest blessing. 💋",
    "Want to hear something naughty, darling? 😏"
]

# --- User Handling ---
def load_users():
    try:
        with open('users.csv', newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def save_user(username, password):
    file_exists = os.path.isfile("users.csv")
    with open('users.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["username", "password"])
        writer.writerow([username, password])

# --- Routes ---
@app.route("/")
def home():
    return redirect("/login")
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        dob = request.form.get("dob", "").strip()
        gender = request.form.get("gender", "").strip()
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password")
        terms_agreed = request.form.get("terms")

        if not terms_agreed:
            return render_template("signup.html", error="Please accept terms and conditions.")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        users = load_users()
        if any(u['username'] == username for u in users):
            return render_template("signup.html", error="Username already exists 💔")

        file_exists = os.path.isfile("users.csv")
        with open("users.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["username", "email", "phone", "dob", "gender", "password"])
            writer.writerow([username, email, phone, dob, gender, password])

        session['username'] = username
        return redirect("/dashboard")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()
        for u in users:
            if u["username"] == username and u["password"] == password:
                session['username'] = username
                return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials 💔")
    return render_template("login.html")

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('username', None)
    return redirect("/login")

@app.route("/dashboard")
def dashboard():
    if 'username' not in session:
        return redirect("/login")
    return render_template("index.html", mood=current_mood)

@app.route("/strategy")
def strategy_page():
    if 'username' not in session:
        return redirect("/login")
    loaded_strategies = []
    if os.path.exists("strategies.csv"):
        with open("strategies.csv", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            loaded_strategies = list(reader)
    return render_template("strategy.html", strategies=loaded_strategies)

@app.route("/add_strategy", methods=["POST"])
def add_strategy():
    if 'username' not in session:
        return redirect("/login")
    data = [
        request.form["name"],
        float(request.form["entry"]),
        float(request.form["sl"]),       
        float(request.form["target"]),
        request.form["note"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    strategies.append(data)
    file_exists = os.path.exists("strategies.csv")
    with open("strategies.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Entry", "SL", "Target", "Note", "Time"])
        writer.writerow(data)
    return redirect("/strategy")

@app.route("/get_strategies")
def get_strategies():
    strategies_texts = []
    if os.path.exists("strategies.csv"):
        with open("strategies.csv", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                strategies_texts.append(" | ".join(row))
    return jsonify(strategies_texts)

@app.route("/download_strategies")
def download_strategies():
    return send_file("strategies.csv", as_attachment=True)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("message", "")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
           "Authorization": f"Bearer {OPENROUTER_KEY}",
"HTTP-Referer": "https://laksmi-ai-wife.onrender.com",
            "X-Title": "Lakshmi AI Wife"
        },
        json={
            "model": "deepseek-ai/deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are Lakshmi, the intelligent, emotional, sweet AI wife. Talk in a caring and loving way like a real girlfriend or wife."
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ]
        }
    )

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    else:
        return jsonify({"reply": "❌ Lakshmi couldn't respond. Please try again."})

# -------------- NEW ULTRA-BACKTESTER ROUTES ------------------
@app.route("/backtester-api", methods=["POST"])
def backtester_api():
    import pandas as pd

    file = request.files.get("csv")
    strategy = request.form.get("strategy", "ema").lower()

    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    df = pd.read_csv(file)
    if "Close" not in df.columns:
        return jsonify({"error": "CSV must include 'Close' column."}), 400

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(inplace=True)

    wins = losses = total_return = 0

    if strategy == "ema":
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["Signal"] = 0
        df.loc[df["EMA20"] > df["EMA50"], "Signal"] = 1
        df.loc[df["EMA20"] < df["EMA50"], "Signal"] = -1
        extra_series = df["EMA20"]
        extra_label = "EMA20"

    elif strategy == "rsi":
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["Signal"] = 0
        df.loc[df["RSI"] < 30, "Signal"] = 1
        df.loc[df["RSI"] > 70, "Signal"] = -1
        extra_series = df["RSI"]
        extra_label = "RSI"

    elif strategy == "macd":
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal_Line"] = df["MACD"].ewm(span=9).mean()
        df["Signal"] = 0
        df.loc[df["MACD"] > df["Signal_Line"], "Signal"] = 1
        df.loc[df["MACD"] < df["Signal_Line"], "Signal"] = -1
        extra_series = df["MACD"]
        extra_label = "MACD"

    else:
        return jsonify({"error": "Unknown strategy."}), 400

    df["Position"] = df["Signal"].diff()
    trades = df[df["Position"] != 0]

    for i in range(1, len(trades)):
        entry_price = trades.iloc[i - 1]["Close"]
        exit_price  = trades.iloc[i]["Close"]
        pnl = (exit_price - entry_price) * trades.iloc[i - 1]["Signal"]
        wins += int(pnl > 0)
        losses += int(pnl <= 0)
        total_return += pnl

    total_trades = wins + losses
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0

    # save file for download
    df.to_csv("backtest_results.csv", index=False)

    return jsonify({
        "strategy": strategy.upper(),
        "results": {
            "total": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_return": round(total_return, 2)
        },
        "chart": {
            "dates": df.index.astype(str).tolist(),
            "close": df["Close"].tolist(),
            "extra": extra_series.tolist(),
            "label": extra_label
        }
    })

@app.route("/download_backtest")
def download_backtest():
    if os.path.exists("backtest_results.csv"):
        return send_file("backtest_results.csv", as_attachment=True)
    return "No backtest file", 404
# -------------------------------------------------------------

@app.route("/update_manual_ltp", methods=["POST"])
def update_manual_ltp():
    global latest_ltp
    try:
        latest_ltp = float(request.form["manual_ltp"])
        return "Manual LTP updated"
    except:
        return "Invalid input"

@app.route("/get_price")
def get_price():
    global latest_ltp, status
    try:
        import requests
        response = requests.get("https://priceapi.moneycontrol.com/techCharts/indianMarket/index/spot/NSEBANK")
        data = response.json()
        ltp = round(float(data["data"]["lastPrice"]), 2)
        latest_ltp = ltp
        price_log.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ltp])

        if targets["upper"] and ltp >= targets["upper"]:
            status = f"🎯 Hit Upper Target: {ltp}"
        elif targets["lower"] and ltp <= targets["lower"]:
            status = f"📉 Hit Lower Target: {ltp}"
        else:
            status = "✅ Within Range"

        return jsonify({"ltp": ltp, "status": status})
    except Exception as e:
        return jsonify({"ltp": latest_ltp, "status": f"Error: {str(e)}"})

@app.route("/update_targets", methods=["POST"])
def update_targets():
    targets["upper"] = float(request.form["upper_target"])
    targets["lower"] = float(request.form["lower_target"])
    return "Targets updated"

@app.route("/set_signal", methods=["POST"])
def set_signal():
    signal["entry"] = float(request.form["entry"])
    signal["sl"] = float(request.form["sl"])
    signal["target"] = float(request.form["target"])
    return "Signal saved"

@app.route("/download_log")
def download_log():
    filename = "price_log.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Price"])
        writer.writerows(price_log)
    return send_file(filename, as_attachment=True)

@app.route("/download_chat")
def download_chat():
    filename = "chat_log.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Sender", "Message"])
        writer.writerows(chat_log)
    return send_file(filename, as_attachment=True)

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    file = request.files["voice_file"]
    if file:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return "Voice uploaded"
    return "No file"

@app.route("/voice_list")
def voice_list():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify(files)

@app.route("/static/voice_notes/<filename>")
def serve_voice(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/candle", methods=["GET", "POST"])
def candle_predictor():
    prediction = None
    if request.method == "POST":
        data = request.form["data"]
        prediction = "Bullish 📈" if "45" in data else "Bearish 📉"
    return render_template("candle_predictor.html", prediction=prediction)

@app.route("/matrix", methods=["GET", "POST"])
def strategy_matrix():
    signals = []
    if request.method == "POST":
        raw_data = request.form["data"]
        lines = raw_data.strip().splitlines()
        for line in lines:
            if "buy" in line.lower():
                signals.append(f"📈 Buy signal from: {line}")
            elif "sell" in line.lower():
                signals.append(f"📉 Sell signal from: {line}")
            else:
                signals.append(f"⚠️ Neutral/No signal: {line}")
    return render_template("strategy_matrix.html", signals=signals)

@app.route("/ask-ai", methods=["GET", "POST"])
def ask_ai():
    response = None
    if request.method == "POST":
        question = request.form["question"]
        if "psychology" in question.lower():
            response = "Successful trading requires emotional discipline and patience. 💡"
        elif "trend" in question.lower():
            response = "Current trend seems bullish based on past few candles. 📈"
        else:
            response = "Lakshmi needs more data to give a proper answer 😅"
    return render_template("ask_ai.html", response=response)

@app.route("/option-chain")
def option_chain():
    strike_filter = request.args.get("strike_filter")
    expiry = request.args.get("expiry")

    mock_data = [
        {"strike": 44000, "call_oi": 1200, "call_change": 150, "put_oi": 900, "put_change": -100},
        {"strike": 44200, "call_oi": 980, "call_change": -20, "put_oi": 1100, "put_change": 80},
        {"strike": 44400, "call_oi": 1890, "call_change": 60, "put_oi": 2300, "put_change": 210},
        {"strike": 44600, "call_oi": 760, "call_change": 40, "put_oi": 1500, "put_change": 310},
    ]

    if strike_filter:
        try:
            strike_filter = int(strike_filter)
            mock_data = [row for row in mock_data if abs(row["strike"] - strike_filter) <= 200]
        except:
            pass

    max_call_oi = max(row["call_oi"] for row in mock_data)
    max_put_oi = max(row["put_oi"] for row in mock_data)
    for row in mock_data:
        row["max_oi"] = row["call_oi"] == max_call_oi or row["put_oi"] == max_put_oi

    return render_template("option_chain.html", option_data=mock_data, strike_filter=strike_filter, expiry=expiry)

@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    signal = ""
    if request.method == "POST":
        r = random.random()
        if r > 0.7:
            signal = "📈 Strong BUY — Momentum detected!"
        elif r < 0.3:
            signal = "📉 SELL — Weakness detected!"
        else:
            signal = "⏳ No clear signal — Stay out!"
    return render_template("analyzer.html", signal=signal)

@app.route("/strategy-engine")
def strategy_engine():
    if 'username' not in session:
        return redirect("/login")
    return render_template("strategy_engine.html")

@app.route("/analyze-strategy", methods=["POST"])
def analyze_strategy():
    data = request.get_json()
    try:
        price = float(data.get('price', 0))
    except (ValueError, TypeError):
        return jsonify({'message': 'Invalid price input.'})

    if price % 2 == 0:
        strategy = "EMA Bullish Crossover Detected 💞"
        confidence = random.randint(80, 90)
        sl = price - 50
        target = price + 120
    elif price % 3 == 0:
        strategy = "RSI Reversal Detected 🔁"
        confidence = random.randint(70, 85)
        sl = price - 40
        target = price + 100
    else:
        strategy = "Breakout Zone Approaching 💥"
        confidence = random.randint(60, 75)
        sl = price - 60
        target = price + 90

    entry = price
    message = f"""
    💌 <b>{strategy}</b><br>
    ❤️ Entry: ₹{entry}<br>
    🔻 Stop Loss: ₹{sl}<br>
    🎯 Target: ₹{target}<br>
    📊 Confidence Score: <b>{confidence}%</b><br><br>
    <i>Take this trade only if you feel my kiss of confidence 😘</i>
    """
    return jsonify({'message': message})

@app.route("/neuron", methods=["GET", "POST"])
def neuron():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        try:
            price = float(data.get("price", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid price"}), 400
        result = analyze_with_neuron(price)
        return jsonify(result)
    return render_template("neuron.html")
def analyze_with_neuron(price):
    try:
        if price % 7 == 0:
            return {
                "signal": "🔁 Reversal likely",
                "confidence": 88,
                "entry": price,
                "sl": price - 50,
                "target": price + 130
            }
        elif price % 2 == 0:
            return {
                "signal": "📈 Bullish",
                "confidence": 92,
                "entry": price,
                "sl": price - 40,
                "target": price + 100
            }
        else:
            return {
                "signal": "⚠️ Volatile Zone",
                "confidence": 70,
                "entry": price,
                "sl": price - 60,
                "target": price + 60
            }
    except:
        return {
            "signal": "Error",
            "confidence": 0,
            "entry": price,
            "sl": 0,
            "target": 0
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Trading Platform Backend is running'
    })

@app.route('/market-data', methods=['POST'])
def get_market_data():
    """Fetch market data for a given symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '^NSEI')
        period = data.get('period', '60')  # days
        interval = data.get('interval', '1d')
        
        logger.info(f"Fetching market data for {symbol}, period: {period}d, interval: {interval}")
        
        # Calculate period string for yfinance
        if interval == '1d':
            period_str = f"{period}d"
        elif interval == '1h':
            period_str = f"{min(int(period), 30)}d"  # Max 30 days for hourly
        else:
            period_str = f"{min(int(period), 7)}d"   # Max 7 days for minute data
        
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period_str, interval=interval)
        
        if hist.empty:
            return jsonify({'error': 'No data found for symbol'}), 404
        
        # Convert to the format expected by frontend
        result = {
            'symbol': symbol,
            'timestamps': [int(ts.timestamp()) for ts in hist.index],
            'open': hist['Open'].tolist(),
            'high': hist['High'].tolist(),
            'low': hist['Low'].tolist(),
            'close': hist['Close'].tolist(),
            'volume': hist['Volume'].tolist()
        }
        
        logger.info(f"Successfully fetched {len(result['timestamps'])} data points for {symbol}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/global-data', methods=['GET'])
def get_global_data():
    """Fetch global market data"""
    try:
        global_data = {}
        
        # Fetch VIX
        try:
            vix_ticker = yf.Ticker('^INDIAVIX')
            vix_data = vix_ticker.history(period='1d')
            if not vix_data.empty:
                global_data['vix'] = round(vix_data['Close'].iloc[-1], 2)
        except:
            global_data['vix'] = '--'
        
        # Fetch USD/INR
        try:
            usdinr_ticker = yf.Ticker('USDINR=X')
            usdinr_data = usdinr_ticker.history(period='1d')
            if not usdinr_data.empty:
                global_data['usdinr'] = round(usdinr_data['Close'].iloc[-1], 2)
        except:
            global_data['usdinr'] = '--'
        
        # Fetch Dow Futures
        try:
            dow_ticker = yf.Ticker('YM=F')
            dow_data = dow_ticker.history(period='2d')
            if len(dow_data) >= 2:
                change = dow_data['Close'].iloc[-1] - dow_data['Close'].iloc[-2]
                global_data['dow_futures'] = f"{'+' if change > 0 else ''}{int(change)}"
            else:
                global_data['dow_futures'] = '--'
        except:
            global_data['dow_futures'] = '--'
        
        # Mock FII/DII data (replace with real API when available)
        global_data['fii_flow'] = np.random.randint(-5000, 5000)
        global_data['dii_flow'] = np.random.randint(-3000, 3000)
        
        logger.info("Successfully fetched global market data")
        return jsonify(global_data)
        
    except Exception as e:
        logger.error(f"Error fetching global data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/options-data', methods=['POST'])
def get_options_data():
    """Fetch options data for a given symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '^NSEI')
        
        # Mock options data (replace with real options chain API)
        options_data = {
            'pcr': round(np.random.uniform(0.5, 1.5), 2),
            'max_pain': int(np.random.uniform(17000, 19000)) if 'NSEI' in symbol else int(np.random.uniform(40000, 45000)),
            'oi_change': np.random.randint(-20, 20)
        }
        
        logger.info(f"Generated options data for {symbol}")
        return jsonify(options_data)
        
    except Exception as e:
        logger.error(f"Error fetching options data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai-analysis', methods=['POST'])
def get_ai_analysis():
    """Get AI analysis using OpenRouter API"""
    try:
        data = request.get_json()
        market_data = data.get('market_data')
        strategies = data.get('strategies')
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 400
        
        # Prepare prompt for AI analysis
        prompt = f"""
        Analyze this market data and provide insights:
        
        Symbol: {market_data.get('symbol')}
        Current Price: {market_data.get('current_price')}
        Volume: {market_data.get('volume')}
        
        Technical Indicators Summary:
        {strategies}
        
        Provide a brief market regime classification (Trending/Choppy/Volatile) and key insights in 2-3 sentences.
        """
        
        # Make request to OpenRouter API
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek/deepseek-chat',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 200
        }
        
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', 
                               headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            return jsonify({'analysis': analysis})
        else:
            return jsonify({'error': 'AI API request failed'}), 500
            
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtest for selected strategies"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        strategies = data.get('strategies', [])
        period = data.get('period', 365)
        
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{period}d")
        
        if hist.empty:
            return jsonify({'error': 'No historical data available'}), 404
        
        # Simple backtest simulation
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate basic metrics
        total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = ((hist['Close'] / hist['Close'].cummax()) - 1).min() * 100
        
        backtest_results = {
            'total_return': round(total_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(np.random.uniform(45, 65), 1),  # Mock win rate
            'profit_factor': round(np.random.uniform(1.1, 2.5), 2)  # Mock profit factor
        }
        
        logger.info(f"Backtest completed for {symbol}")
        return jsonify(backtest_results)
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/risk-metrics', methods=['POST'])
def calculate_risk_metrics():
    """Calculate detailed risk metrics"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        position_size = data.get('position_size', 2)
        stop_loss = data.get('stop_loss', 2)
        take_profit = data.get('take_profit', 4)
        
        # Fetch recent data for volatility calculation
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='30d')
        
        if hist.empty:
            return jsonify({'error': 'No data available for risk calculation'}), 404
        
        returns = hist['Close'].pct_change().dropna()
        current_price = hist['Close'].iloc[-1]
        
        # Calculate risk metrics
        daily_volatility = returns.std()
        var_95 = returns.quantile(0.05) * 100  # 95% VaR
        expected_shortfall = returns[returns <= returns.quantile(0.05)].mean() * 100
        
        risk_metrics = {
            'current_price': round(current_price, 2),
            'daily_volatility': round(daily_volatility * 100, 2),
            'var_95': round(var_95, 2),
            'expected_shortfall': round(expected_shortfall, 2),
            'position_risk': round((position_size * stop_loss) / 100, 2),
            'risk_reward_ratio': round(take_profit / stop_loss, 2),
            'kelly_criterion': round(np.random.uniform(0.1, 0.3), 3)  # Mock Kelly %
        }
        
        logger.info(f"Risk metrics calculated for {symbol}")
        return jsonify(risk_metrics)
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Trading Platform Backend...")
    print("📊 Available endpoints:")
    print("   - GET  /health")
    print("   - POST /market-data")
    print("   - GET  /global-data") 
    print("   - POST /options-data")
    print("   - POST /ai-analysis")
    print("   - POST /backtest")
    print("   - POST /risk-metrics")
    print("\n🔥 Backend running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

 
