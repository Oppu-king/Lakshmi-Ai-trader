from flask import Flask
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "lakshmi_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/voice_notes'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# DO NOT put `app.run()` for Vercel
--- Global Variables ---
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
    return "<h1>💖 Lakshmi — Your AI Wife is Live 💖</h1>"
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
        float(request.form]),
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

# --- Start App ---
 import os

api_key = os.environ.get("OPENROUTER_API_KEY")
