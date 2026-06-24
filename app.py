import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import sqlite3
import tempfile
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

st.set_page_config(page_title="Weather Intelligence Pro", page_icon="🌦️", layout="wide")

API_KEY = "29605a0bdb9126e27fa0e15c41a66ac7"

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(14,165,233,0.26), transparent 34%),
        radial-gradient(circle at bottom right, rgba(168,85,247,0.20), transparent 34%),
        linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.13);
}
.hero {
    padding: 34px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 28px 80px rgba(0,0,0,0.40);
    margin-bottom: 26px;
    animation: fadeUp 0.8s ease-in-out;
}
.hero-title {
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: #cbd5e1; font-size: 17px; margin-top: 8px; }
.weather-main-card {
    padding: 28px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(56,189,248,0.20), rgba(139,92,246,0.12)), rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 24px 70px rgba(0,0,0,0.35);
    margin-bottom: 18px;
    animation: fadeUp 0.7s ease-in-out;
}
.weather-city { font-size: 28px; font-weight: 850; }
.weather-temp { font-size: 74px; font-weight: 950; line-height: 1; color: #38bdf8; }
.weather-condition { color: #e2e8f0; font-size: 20px; margin-top: 10px; }
.weather-updated { color: #94a3b8; font-size: 14px; margin-top: 8px; }
.glass-card {
    padding: 22px;
    border-radius: 22px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
    transition: 0.25s ease;
    min-height: 125px;
    margin-bottom: 15px;
}
.glass-card:hover { transform: translateY(-5px); border-color: rgba(56,189,248,0.45); }
.card-label { color: #cbd5e1; font-size: 15px; font-weight: 700; }
.card-value { color: #38bdf8; font-size: 30px; font-weight: 900; margin-top: 8px; }
.summary-box {
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(56,189,248,0.12));
    border: 1px solid rgba(34,197,94,0.25);
    color: #e5e7eb;
}
.risk-box {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(251,191,36,0.13), rgba(244,63,94,0.10));
    border: 1px solid rgba(251,191,36,0.25);
}
.forecast-card {
    padding: 18px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    text-align: center;
    box-shadow: 0 14px 40px rgba(0,0,0,0.22);
    min-height: 150px;
}
.forecast-date { font-weight: 800; color: #e5e7eb; }
.forecast-temp { font-size: 30px; font-weight: 900; color: #38bdf8; margin-top: 8px; }
.forecast-condition { color: #cbd5e1; font-size: 13px; margin-top: 5px; }
[data-testid="stMetric"] {
    padding: 18px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 14px 40px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] { color: #38bdf8; font-weight: 800; }
.stButton > button {
    width: 100%; border-radius: 15px; padding: 0.75rem 1rem; border: none; font-weight: 850;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6); color: white;
    box-shadow: 0 12px 35px rgba(56,189,248,0.25);
}
.stButton > button:hover { transform: scale(1.02); }
.about-box {
    padding: 22px;
    border-radius: 22px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    color: #dbeafe;
}
@keyframes fadeUp { from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: translateY(0);} }
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
def create_database():
    conn = sqlite3.connect("weather.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            temperature REAL,
            feels_like REAL,
            humidity REAL,
            pressure REAL,
            wind_speed REAL,
            condition TEXT,
            source TEXT,
            searched_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_weather(city, temp, feels_like, humidity, pressure, wind, condition, source):
    conn = sqlite3.connect("weather.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO weather_history
        (city, temperature, feels_like, humidity, pressure, wind_speed, condition, source, searched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        city, temp, feels_like, humidity, pressure, wind, condition, source,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect("weather.db")
    df = pd.read_sql_query("SELECT * FROM weather_history ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_history():
    conn = sqlite3.connect("weather.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM weather_history")
    conn.commit()
    conn.close()

create_database()

# ---------------- SESSION STATE ----------------
def init_state():
    defaults = {
        "selected_weather_data": None,
        "selected_forecast_data": None,
        "selected_aqi_data": None,
        "selected_city_name": None,
        "selected_lat": None,
        "selected_lon": None,
        "selected_recommendation": None,
        "selected_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ---------------- API FUNCTIONS ----------------
def search_locations(place):
    locations = []
    try:
        url = f"https://api.openweathermap.org/geo/1.0/direct?q={place}&limit=5&appid={API_KEY}"
        data = requests.get(url, timeout=10).json()
        if isinstance(data, list):
            for loc in data:
                if "lat" in loc and "lon" in loc:
                    locations.append({
                        "name": loc.get("name", place),
                        "state": loc.get("state", ""),
                        "country": loc.get("country", ""),
                        "lat": float(loc["lat"]),
                        "lon": float(loc["lon"]),
                        "source": "OpenWeather Geocoding"
                    })
    except Exception:
        pass

    try:
        osm_url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 5, "addressdetails": 1}
        headers = {"User-Agent": "WeatherDashboardApp/1.0"}
        data = requests.get(osm_url, params=params, headers=headers, timeout=10).json()
        if isinstance(data, list):
            for loc in data:
                address = loc.get("address", {})
                name = address.get("city") or address.get("town") or address.get("village") or address.get("hamlet") or loc.get("display_name", place).split(",")[0]
                locations.append({
                    "name": name,
                    "state": address.get("state", ""),
                    "country": address.get("country", ""),
                    "lat": float(loc["lat"]),
                    "lon": float(loc["lon"]),
                    "source": "OpenStreetMap"
                })
    except Exception:
        pass

    unique = []
    seen = set()
    for loc in locations:
        key = (round(loc["lat"], 3), round(loc["lon"], 3))
        if key not in seen:
            unique.append(loc)
            seen.add(key)
    return unique


def get_current_weather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        data = requests.get(url, timeout=10).json()
        if data.get("cod") == 200:
            return {
                "ok": True,
                "source": "OpenWeather Current API",
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "condition": data["weather"][0]["description"],
                "main_condition": data["weather"][0]["main"],
                "icon": data["weather"][0]["icon"],
                "sunrise": data["sys"]["sunrise"],
                "sunset": data["sys"]["sunset"],
                "timezone": data["timezone"]
            }
    except Exception:
        pass
    return {"ok": False}


def get_forecast(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        return requests.get(url, timeout=10).json()
    except Exception:
        return {}


def get_aqi(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        return requests.get(url, timeout=10).json()
    except Exception:
        return {}


def convert_time(timestamp, timezone_offset):
    if timestamp is None:
        return "Not available"
    return datetime.utcfromtimestamp(timestamp + timezone_offset).strftime("%I:%M %p")

# ---------------- LOTTIE ----------------
def load_lottie_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None


def show_weather_animation(condition):
    if not LOTTIE_AVAILABLE:
        st.caption("Install streamlit-lottie for animations: py -m pip install streamlit-lottie")
        return
    condition = str(condition).lower()
    if "rain" in condition or "drizzle" in condition:
        url = "https://assets2.lottiefiles.com/packages/lf20_jmBauI.json"
    elif "snow" in condition:
        url = "https://assets9.lottiefiles.com/packages/lf20_WtPCZs.json"
    elif "cloud" in condition:
        url = "https://assets10.lottiefiles.com/packages/lf20_V9t630.json"
    elif "thunder" in condition:
        url = "https://assets2.lottiefiles.com/packages/lf20_KUFdS6.json"
    else:
        url = "https://assets9.lottiefiles.com/packages/lf20_xlky4kvh.json"
    animation = load_lottie_url(url)
    if animation:
        st_lottie(animation, height=150)

# ---------------- HELPERS ----------------
def clean_history_data(df):
    if df.empty:
        return df
    numeric_cols = ["temperature", "feels_like", "humidity", "pressure", "wind_speed"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["searched_at"] = pd.to_datetime(df["searched_at"], errors="coerce")
    df = df.dropna(subset=["temperature", "humidity", "pressure", "wind_speed"])
    return df


def get_aqi_label(aqi):
    return {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}.get(aqi, "Not Available")


def get_condition_emoji(condition):
    condition = str(condition).lower()
    if "rain" in condition or "drizzle" in condition:
        return "🌧️"
    if "snow" in condition:
        return "❄️"
    if "cloud" in condition:
        return "☁️"
    if "thunder" in condition:
        return "⛈️"
    if "mist" in condition or "fog" in condition or "haze" in condition:
        return "🌫️"
    return "☀️"


def get_weather_score(temp, humidity, wind, aqi=None):
    score = 100
    if temp < 5 or temp > 40:
        score -= 30
    elif temp < 12 or temp > 35:
        score -= 18
    if humidity > 85:
        score -= 18
    elif humidity > 70:
        score -= 10
    if wind > 15:
        score -= 20
    elif wind > 10:
        score -= 10
    if aqi == 5:
        score -= 30
    elif aqi == 4:
        score -= 22
    elif aqi == 3:
        score -= 12
    return max(score, 0)


def get_risk_level(score):
    if score >= 80:
        return "Excellent", "✅"
    if score >= 60:
        return "Good", "🟢"
    if score >= 40:
        return "Moderate Risk", "🟡"
    return "High Risk", "🔴"


def generate_weather_summary(city, weather, aqi):
    temp = weather["temperature"]
    feels = weather["feels_like"]
    humidity = weather["humidity"]
    wind = weather["wind_speed"]
    condition = str(weather["condition"]).title()
    aqi_text = get_aqi_label(aqi) if aqi else "not available"
    if temp >= 35:
        temp_line = "The temperature is on the hotter side, so outdoor activity should be planned carefully."
    elif temp <= 10:
        temp_line = "The weather is cold, so warm clothing is recommended."
    else:
        temp_line = "The temperature is comfortable for regular outdoor movement."
    humidity_line = "Humidity is high, so the weather may feel sticky or tiring." if humidity >= 80 else "Humidity is within a manageable range."
    wind_line = "Wind speed is high, so biking or exposed-area travel should be avoided." if wind >= 12 else "Wind speed is normal for general travel."
    return f"{city} is currently experiencing {condition} with a temperature of {temp}°C and feels like {feels}°C. {temp_line} {humidity_line} {wind_line} Air quality is {aqi_text}."


def get_travel_recommendation(temp, humidity, wind, condition, aqi=None):
    condition = str(condition).lower()
    tips = []
    if temp <= 5:
        tips.append("Very cold weather. Carry heavy jacket, gloves and warm layers.")
    elif temp <= 12:
        tips.append("Cold weather. Carry jacket or hoodie.")
    elif temp >= 40:
        tips.append("Extreme heat. Avoid afternoon travel and stay hydrated.")
    elif temp >= 34:
        tips.append("Hot weather. Carry water and avoid long direct sunlight.")
    else:
        tips.append("Temperature is comfortable for normal travel.")
    if "rain" in condition or "drizzle" in condition:
        tips.append("Carry umbrella or raincoat.")
    elif "snow" in condition:
        tips.append("Snow condition detected. Wear waterproof shoes.")
    elif "thunder" in condition:
        tips.append("Avoid trekking, biking and open areas.")
    if humidity >= 80:
        tips.append("High humidity may make weather feel tiring.")
    if wind >= 12:
        tips.append("Strong wind. Bike riding is not recommended.")
    if aqi is not None and aqi >= 4:
        tips.append("Poor air quality. Avoid long outdoor activity.")
    return " ".join(tips)


def _is_risky_conditions(temp, wind, condition_lower):
    """Check if conditions are risky for outdoor activities."""
    return temp <= 5 or wind >= 12 or "rain" in condition_lower or "thunder" in condition_lower


def _is_bike_unsafe(wind, condition_lower):
    """Check if conditions are unsafe for biking."""
    return wind >= 12 or "rain" in condition_lower or "thunder" in condition_lower


def _has_rain(condition_lower):
    """Check if rain is present in condition."""
    return "rain" in condition_lower or "drizzle" in condition_lower


def _answer_trek_question(city):
    """Generate answer for trek-related questions."""
    return f"For {city}, trekking looks manageable. Carry water, jacket, power bank, snacks and check local trail conditions before starting."


def _answer_bike_question(city):
    """Generate answer for bike-related questions."""
    return f"Bike ride looks okay in {city}. Wear helmet, jacket, gloves and avoid late-night riding."


def _answer_clothes_question(temp, city):
    """Generate answer for clothing-related questions."""
    if temp <= 15:
        return f"Yes, carry a jacket or hoodie. Current temperature in {city} is {temp}°C."
    return "Light clothes are okay, but carry a light layer if you are travelling in hills or staying out late."


def chatbot_answer(question, weather=None, city="this location", aqi=None):
    q = question.lower().strip()
    if weather is None:
        return "Search live weather first, then ask me for travel or weather advice."
    if not q:
        return "Ask me something like: Can I go for trek? Should I carry jacket? Is bike ride safe?"
    
    temp = weather["temperature"]
    humidity = weather["humidity"]
    wind = weather["wind_speed"]
    condition = weather["condition"]
    condition_lower = condition.lower()
    
    # Trek/hike questions
    if any(keyword in q for keyword in ["trek", "hike", "hiking"]):
        if _is_risky_conditions(temp, wind, condition_lower):
            return f"For {city}, trekking is not recommended right now because current weather may be risky due to cold, rain, thunder or strong wind."
        return _answer_trek_question(city)
    
    # Bike/ride questions
    if any(keyword in q for keyword in ["bike", "ride", "scooty"]):
        if _is_bike_unsafe(wind, condition_lower):
            return f"Bike ride is not recommended in {city} right now because wind/rain/thunder can make roads unsafe."
        return _answer_bike_question(city)
    
    # Clothing questions
    if any(keyword in q for keyword in ["jacket", "wear", "clothes"]):
        return _answer_clothes_question(temp, city)
    
    # Rain questions
    if any(keyword in q for keyword in ["umbrella", "raincoat", "rain"]):
        if _has_rain(condition_lower):
            return f"Yes, carry an umbrella or raincoat. Current condition in {city} is {condition}."
        return "Rain is not clearly detected in current condition, but check the forecast section before going out."
    
    # AQI questions
    if any(keyword in q for keyword in ["aqi", "pollution", "air"]):
        return f"AQI in {city} is {aqi} - {get_aqi_label(aqi)}." if aqi else "AQI data is not available for this location."
    
    # Safety/travel questions
    if any(keyword in q for keyword in ["safe", "travel", "good"]):
        return get_travel_recommendation(temp, humidity, wind, condition, aqi)
    
    return generate_weather_summary(city, weather, aqi)




def build_daily_forecast_cards(forecast):
    if not forecast or forecast.get("cod") != "200":
        return None
    rows = []
    for item in forecast["list"]:
        dt = pd.to_datetime(item["dt_txt"])
        rows.append({
            "date": dt.date(),
            "datetime": dt,
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "wind": item["wind"]["speed"],
            "condition": item["weather"][0]["description"]
        })
    df = pd.DataFrame(rows)
    daily = df.groupby("date").agg({"temp": "mean", "humidity": "mean", "wind": "mean", "condition": "first"}).reset_index()
    return daily.head(5)


def create_weather_cards(weather, aqi):
    score = get_weather_score(weather["temperature"], weather["humidity"], weather["wind_speed"], aqi)
    risk, risk_icon = get_risk_level(score)
    cards = [
        ("🌡️ Temperature", f"{weather['temperature']} °C"),
        ("🤒 Feels Like", f"{weather['feels_like']} °C"),
        ("💧 Humidity", f"{weather['humidity']}%"),
        ("💨 Wind Speed", f"{weather['wind_speed']} m/s"),
        ("🔽 Pressure", f"{weather['pressure']} hPa"),
        ("🌬️ AQI", f"{aqi} - {get_aqi_label(aqi)}" if aqi else "Not Available"),
        ("🏆 Score", f"{score}/100"),
        (f"{risk_icon} Risk Level", risk),
    ]
    cols = st.columns(4)
    for index, (label, value) in enumerate(cards):
        with cols[index % 4]:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- PDF ----------------
def create_pdf_report(city, weather, aqi, recommendation, summary, forecast=None):
    if not PDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 22)
    pdf.cell(0, 14, "Weather Intelligence Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "Professional Weather Summary and Travel Advisory", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Location: {city}", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Weather Source: {weather['source']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, "Current Weather", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Temperature: {weather['temperature']} C", ln=True)
    pdf.cell(0, 8, f"Feels Like: {weather['feels_like']} C", ln=True)
    pdf.cell(0, 8, f"Humidity: {weather['humidity']}%", ln=True)
    pdf.cell(0, 8, f"Pressure: {weather['pressure']} hPa", ln=True)
    pdf.cell(0, 8, f"Wind Speed: {weather['wind_speed']} m/s", ln=True)
    pdf.cell(0, 8, f"Condition: {str(weather['condition']).title()}", ln=True)
    pdf.cell(0, 8, f"AQI: {aqi} - {get_aqi_label(aqi)}" if aqi else "AQI: Not Available", ln=True)
    score = get_weather_score(weather["temperature"], weather["humidity"], weather["wind_speed"], aqi)
    risk, _ = get_risk_level(score)
    pdf.cell(0, 8, f"Weather Score: {score}/100", ln=True)
    pdf.cell(0, 8, f"Risk Level: {risk}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, "AI Weather Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, "Travel Recommendation", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, recommendation)
    daily = build_daily_forecast_cards(forecast)
    if daily is not None:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, "5-Day Forecast Summary", ln=True)
        pdf.set_font("Arial", "", 10)
        for _, row in daily.iterrows():
            line = f"{row['date']} | Avg Temp: {row['temp']:.1f} C | Humidity: {row['humidity']:.0f}% | Wind: {row['wind']:.1f} m/s | {str(row['condition']).title()}"
            pdf.multi_cell(0, 7, line)
    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6, "Developed by Nakul Bidhuri using Python, Streamlit, SQLite, OpenWeather API, Plotly and Machine Learning.")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🌦️ Weather Intelligence Pro")
st.sidebar.caption("Premium AI-powered weather dashboard")
page = st.sidebar.radio(
    "Choose Section",
    ["🌦️ Dashboard", "🌍 Compare Cities", "📊 Analytics", "🤖 ML Prediction", "🧪 Anomaly Detection",  "🗄️ History"]
)
st.sidebar.markdown("---")
st.sidebar.info(" Built by  Nakul Bidhuri ❤️")

# ---------------- DASHBOARD ----------------
if page == "🌦️ Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Weather Intelligence Pro</div>
        <div class="hero-subtitle">Real-time weather, AQI, premium insights, AI summary, chatbot, maps, reports and machine learning.</div>
    </div>
    """, unsafe_allow_html=True)
    place = st.text_input("Search any city/place", placeholder="Example: Kaza, Spiti, Tungnath, Delhi, London")
    selected_data = None
    if place:
        locations = search_locations(place)
        if locations:
            options = []
            for loc in locations:
                options.append(f"{loc['name']}, {loc['state']}, {loc['country']} | Lat: {loc['lat']:.4f}, Lon: {loc['lon']:.4f} | {loc['source']}")
            selected_option = st.selectbox("Select correct location", options)
            selected_data = locations[options.index(selected_option)]
        else:
            st.warning("No location found. Try adding state/country.")

    if st.button("Get Current Weather"):
        if selected_data is None:
            st.error("Please search and select a location first.")
            st.stop()
        lat = selected_data["lat"]
        lon = selected_data["lon"]
        city = f"{selected_data['name']}, {selected_data['state']}, {selected_data['country']}"
        loading = st.empty()
        progress = st.progress(0)
        loading.info("🔍 Finding accurate location...")
        progress.progress(20)
        loading.info("🌎 Connecting to weather server...")
        weather = get_current_weather(lat, lon)
        progress.progress(45)
        loading.info("🌬️ Fetching air quality data...")
        aqi_data = get_aqi(lat, lon)
        progress.progress(65)
        loading.info("📅 Loading forecast data...")
        forecast = get_forecast(lat, lon)
        progress.progress(85)
        loading.info("📊 Generating smart insights...")
        progress.progress(100)
        loading.empty()
        if not weather["ok"]:
            st.error("Weather data not available.")
            st.stop()
        aqi = aqi_data["list"][0]["main"]["aqi"] if "list" in aqi_data else None
        recommendation = get_travel_recommendation(weather["temperature"], weather["humidity"], weather["wind_speed"], weather["condition"], aqi)
        summary = generate_weather_summary(city, weather, aqi)
        save_weather(city, weather["temperature"], weather["feels_like"], weather["humidity"], weather["pressure"], weather["wind_speed"], weather["condition"], weather["source"])
        st.session_state.selected_weather_data = weather
        st.session_state.selected_forecast_data = forecast
        st.session_state.selected_aqi_data = aqi
        st.session_state.selected_city_name = city
        st.session_state.selected_lat = lat
        st.session_state.selected_lon = lon
        st.session_state.selected_recommendation = recommendation
        st.session_state.selected_summary = summary

    if st.session_state.selected_weather_data is not None:
        weather = st.session_state.selected_weather_data
        forecast = st.session_state.selected_forecast_data
        aqi = st.session_state.selected_aqi_data
        city = st.session_state.selected_city_name
        lat = st.session_state.selected_lat
        lon = st.session_state.selected_lon
        recommendation = st.session_state.selected_recommendation
        summary = st.session_state.selected_summary
        emoji = get_condition_emoji(weather["condition"])
        updated_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
        score = get_weather_score(weather["temperature"], weather["humidity"], weather["wind_speed"], aqi)
        risk, risk_icon = get_risk_level(score)
        st.markdown(f"""
        <div class="weather-main-card">
            <div class="weather-city">📍 {city}</div>
            <div class="weather-updated">⏰ Updated: {updated_time} • Source: {weather['source']}</div>
            <div class="weather-temp">{weather['temperature']}°C</div>
            <div class="weather-condition">{emoji} {str(weather['condition']).title()} • Feels Like {weather['feels_like']}°C</div>
            <div class="weather-updated">Risk Level: {risk_icon} {risk} • Weather Score: {score}/100</div>
        </div>
        """, unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Map", "Forecast", "AI Chatbot", "PDF Report"])
        with tab1:
            col_anim, col_summary = st.columns([1, 2])
            with col_anim:
                show_weather_animation(weather["condition"])
                if weather["icon"]:
                    st.image(f"https://openweathermap.org/img/wn/{weather['icon']}@4x.png", width=120)
            with col_summary:
                st.markdown(f"""<div class="summary-box"><h3>🤖 AI Weather Summary</h3><p>{summary}</p></div>""", unsafe_allow_html=True)
            st.markdown("### Weather Details")
            create_weather_cards(weather, aqi)
            st.markdown("### Smart Recommendation")
            st.markdown(f"""<div class="risk-box"><b>Travel Advice:</b><br>{recommendation}</div>""", unsafe_allow_html=True)
            history_df = clean_history_data(load_history())
            if not history_df.empty and len(history_df) > 2:
                avg_temp = history_df["temperature"].mean()
                diff = weather["temperature"] - avg_temp
                insight = f"Current temperature is {abs(diff):.1f}°C {'higher' if diff > 0 else 'lower'} than your average searched temperature."
                st.info(f"📌 Insight: {insight}")
        with tab2:
            st.subheader("🗺️ Interactive Weather Map")
            if FOLIUM_AVAILABLE:
                m = folium.Map(location=[lat, lon], zoom_start=10)
                popup_text = f"""<b>{city}</b><br>Temp: {weather['temperature']} °C<br>Feels Like: {weather['feels_like']} °C<br>Humidity: {weather['humidity']}%<br>Wind: {weather['wind_speed']} m/s<br>AQI: {aqi if aqi else 'N/A'}<br>Condition: {weather['condition']}"""
                folium.Marker([lat, lon], popup=popup_text, tooltip=city).add_to(m)
                folium.Circle(location=[lat, lon], radius=4000, color="blue", fill=True, fill_opacity=0.15).add_to(m)
                st_folium(m, width=1200, height=500)
            else:
                st.warning("Install map packages: py -m pip install folium streamlit-folium")
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
        with tab3:
            st.subheader("📅 5-Day Forecast Cards")
            daily = build_daily_forecast_cards(forecast)
            if daily is not None:
                cols = st.columns(len(daily))
                for i, row in daily.iterrows():
                    with cols[i]:
                        icon = get_condition_emoji(row["condition"])
                        st.markdown(f"""
                        <div class="forecast-card">
                            <div class="forecast-date">{row['date']}</div>
                            <div style="font-size:34px; margin-top:8px;">{icon}</div>
                            <div class="forecast-temp">{row['temp']:.1f}°C</div>
                            <div class="forecast-condition">{str(row['condition']).title()}</div>
                            <div class="forecast-condition">Humidity {row['humidity']:.0f}%</div>
                            <div class="forecast-condition">Wind {row['wind']:.1f} m/s</div>
                        </div>
                        """, unsafe_allow_html=True)
                forecast_df = pd.DataFrame([{"DateTime": item["dt_txt"], "Temperature": item["main"]["temp"], "Feels Like": item["main"]["feels_like"], "Humidity": item["main"]["humidity"], "Wind Speed": item["wind"]["speed"], "Condition": item["weather"][0]["description"]} for item in forecast["list"]])
                st.markdown("### Forecast Chart")
                fig = px.line(forecast_df, x="DateTime", y="Temperature", markers=True, title="5-Day Temperature Forecast", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Show full forecast table"):
                    st.dataframe(forecast_df, use_container_width=True)
            else:
                st.info("Forecast data not available.")
        with tab4:
            st.subheader("🤖 AI Weather Chatbot")
            st.caption("Ask: Can I go for trek? Should I carry jacket? Is bike ride safe? How is AQI?")
            user_question = st.text_input("Ask something", placeholder="Example: Can I go for bike ride today?")
            colq1, colq2, colq3 = st.columns(3)
            with colq1:
                if st.button("Can I go for trek?"):
                    user_question = "Can I go for trek?"
            with colq2:
                if st.button("Should I carry jacket?"):
                    user_question = "Should I carry jacket?"
            with colq3:
                if st.button("Is bike ride safe?"):
                    user_question = "Is bike ride safe?"
            if st.button("Ask Chatbot"):
                answer = chatbot_answer(user_question, weather, city, aqi)
                st.success(answer)
        with tab5:
            st.subheader("📄 Professional PDF Weather Report")
            st.write("This report includes current weather, AQI, weather score, risk level, AI weather summary, travel recommendation and 5-day forecast summary.")
            if PDF_AVAILABLE:
                pdf_path = create_pdf_report(city, weather, aqi, recommendation, summary, forecast)
                if pdf_path:
                    with open(pdf_path, "rb") as file:
                        st.download_button("Download Professional PDF Report", data=file, file_name="weather_intelligence_report.pdf", mime="application/pdf")
            else:
                st.warning("Install PDF package: py -m pip install fpdf")
        if st.button("Clear Current Result"):
            for key in ["selected_weather_data", "selected_forecast_data", "selected_aqi_data", "selected_city_name", "selected_lat", "selected_lon", "selected_recommendation", "selected_summary"]:
                st.session_state[key] = None
            st.rerun()

elif page == "🌍 Compare Cities":
    st.title("🌍 Compare Cities")
    cities_input = st.text_input("Enter places separated by comma", placeholder="Delhi, Mumbai, Kaza, London")
    if st.button("Compare"):
        if not cities_input:
            st.error("Enter cities first.")
            st.stop()
        rows = []
        for city_name in [c.strip() for c in cities_input.split(",")]:
            locs = search_locations(city_name)
            if locs:
                loc = locs[0]
                weather = get_current_weather(loc["lat"], loc["lon"])
                aqi_data = get_aqi(loc["lat"], loc["lon"])
                aqi = aqi_data["list"][0]["main"]["aqi"] if "list" in aqi_data else None
                if weather["ok"]:
                    rows.append({"City": f"{loc['name']}, {loc['country']}", "Temperature": weather["temperature"], "Humidity": weather["humidity"], "Wind": weather["wind_speed"], "AQI": aqi if aqi else 0, "Score": get_weather_score(weather["temperature"], weather["humidity"], weather["wind_speed"], aqi)})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            fig = px.bar(df, x="City", y="Temperature", title="Temperature Comparison", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.bar(df, x="City", y="Score", title="Weather Score Comparison", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("No weather data found.")

elif page == "📊 Analytics":
    st.title("📊 Data Science Analytics")
    history_df = clean_history_data(load_history())
    if history_df.empty:
        st.info("Search weather first.")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Searches", len(history_df))
    c2.metric("Places Tracked", history_df["city"].nunique())
    c3.metric("Average Temp", f"{history_df['temperature'].mean():.2f} °C")
    c4.metric("Highest Temp", f"{history_df['temperature'].max():.2f} °C")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Lowest Temp", f"{history_df['temperature'].min():.2f} °C")
    c6.metric("Avg Humidity", f"{history_df['humidity'].mean():.2f}%")
    c7.metric("Avg Wind", f"{history_df['wind_speed'].mean():.2f} m/s")
    c8.metric("Avg Pressure", f"{history_df['pressure'].mean():.2f} hPa")
    fig = px.line(history_df.sort_values("id"), x="searched_at", y="temperature", color="city", markers=True, title="Temperature Trend", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    corr = history_df[["temperature", "feels_like", "humidity", "pressure", "wind_speed"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap", template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)
    city_count = history_df["city"].value_counts().reset_index()
    city_count.columns = ["City", "Search Count"]
    fig_city = px.bar(city_count, x="City", y="Search Count", title="Most Searched Places", template="plotly_dark")
    st.plotly_chart(fig_city, use_container_width=True)

elif page == "🤖 ML Prediction":
    st.title("🤖 ML Temperature Prediction")
    history_df = clean_history_data(load_history())
    if len(history_df) < 10:
        st.warning("Need at least 10 saved searches for ML prediction.")
        st.stop()
    history_df = history_df.sort_values("id")
    history_df["Search_Number"] = range(1, len(history_df) + 1)
    X = history_df[["Search_Number", "humidity", "wind_speed", "pressure"]]
    y = history_df["temperature"]
    model_choice = st.selectbox("Choose ML Model", ["Random Forest", "XGBoost"] if XGBOOST_AVAILABLE else ["Random Forest"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    if model_choice == "XGBoost" and XGBOOST_AVAILABLE:
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f} °C")
    c2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
    humidity = st.slider("Humidity", 0, 100, 50)
    wind = st.slider("Wind Speed", 0.0, 30.0, 3.0)
    pressure = st.slider("Pressure", 850, 1100, 1010)
    prediction = model.predict([[len(history_df) + 1, humidity, wind, pressure]])[0]
    st.success(f"Predicted Temperature: {prediction:.2f} °C")
    st.warning("This is ML prediction, not actual current temperature.")
    importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    fig = px.bar(importance, x="Feature", y="Importance", title="Feature Importance", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🧪 Anomaly Detection":
    st.title("🧪 Weather Anomaly Detection")
    history_df = clean_history_data(load_history())
    if len(history_df) < 10:
        st.warning("Need at least 10 saved searches.")
        st.stop()
    features = history_df[["temperature", "humidity", "pressure", "wind_speed"]]
    model = IsolationForest(contamination=0.15, random_state=42)
    history_df["Anomaly"] = model.fit_predict(features)
    history_df["Status"] = history_df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
    st.dataframe(history_df, use_container_width=True)
    fig = px.scatter(history_df, x="humidity", y="temperature", color="Status", size="wind_speed", hover_data=["city", "condition"], title="Anomaly Detection", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)




elif page == "🗄️ History":
    st.title("🗄️ Weather History")
    history_df = load_history()
    if history_df.empty:
        st.info("No history found.")
    else:
        st.dataframe(history_df, use_container_width=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="weather_history.csv", mime="text/csv")
        if st.button("Clear History"):
            clear_history()
            st.success("History cleared.")
            st.rerun()

# footer
st.markdown(
    '<div class="footer">     Built by  Nakul Bidhuri ❤️  </div>',
    unsafe_allow_html=True,
)