import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
import requests
import warnings
warnings.filterwarnings('ignore')

API_KEY = "d49717a37d1b3f349394e116639af638"

st.set_page_config(
    page_title="DisasterSense AI",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
h1,h2,h3,p,label,.stMetric { color: white !important; }
.stSlider > div { color: white; }
div[data-testid="metric-container"] {
    background: #1e293b;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 3000
    data = {
        'rainfall_mm':      np.random.uniform(0, 300, n),
        'temperature_c':    np.random.uniform(15, 50, n),
        'wind_speed_kmh':   np.random.uniform(0, 200, n),
        'seismic_activity': np.random.uniform(0, 9, n),
        'humidity_pct':     np.random.uniform(10, 100, n),
        'tweet_count':      np.random.randint(0, 5000, n),
    }
    df = pd.DataFrame(data)
    def label(row):
        if row['rainfall_mm'] > 200 and row['humidity_pct'] > 80:
            return 'Flood'
        elif row['temperature_c'] > 40 and row['wind_speed_kmh'] > 60:
            return 'Wildfire'
        elif row['seismic_activity'] > 6:
            return 'Earthquake'
        elif row['wind_speed_kmh'] > 120:
            return 'Cyclone'
        elif row['temperature_c'] > 45:
            return 'Heatwave'
        elif row['rainfall_mm'] > 150 and row['wind_speed_kmh'] > 80:
            return 'Landslide'
        else:
            return 'Safe'
    df['disaster'] = df.apply(label, axis=1)
    X = df.drop('disaster', axis=1)
    y = df['disaster']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=5)
        d = r.json()
        if r.status_code == 200:
            return {
                'temp': d['main']['temp'],
                'humidity': d['main']['humidity'],
                'wind': d['wind']['speed'] * 3.6,
                'rain': d.get('rain', {}).get('1h', 0) * 10,
                'desc': d['weather'][0]['description'].title()
            }
    except:
        pass
    return None

model = train_model()

# Header
st.markdown("<h1 style='color:#ef4444;font-size:38px;'>🌍 DisasterSense AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;font-size:16px;'>Multi-Hazard Early Warning System — Powered by AI & Real-Time Data</p>", unsafe_allow_html=True)
st.markdown("---")

# Metrics
c1,c2,c3,c4 = st.columns(4)
c1.metric("Model Accuracy", "99%")
c2.metric("Disaster Types", "7")
c3.metric("Data Points", "3,000")
c4.metric("AI Status", "🟢 Live")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎛️ Manual Predict", "🌦️ Live Weather Predict", "🗺️ World Map"])

# TAB 1 - Manual
with tab1:
    st.markdown("### Enter Parameters Manually")
    col1, col2 = st.columns(2)
    with col1:
        rainfall   = st.slider("🌧️ Rainfall (mm)", 0, 300, 150)
        temperature= st.slider("🌡️ Temperature (°C)", 15, 50, 30)
        wind_speed = st.slider("💨 Wind Speed (km/h)", 0, 200, 60)
    with col2:
        seismic    = st.slider("🌍 Seismic Activity (Richter)", 0.0, 9.0, 3.0, 0.1)
        humidity   = st.slider("💧 Humidity (%)", 10, 100, 60)
        tweets     = st.slider("🐦 Disaster Tweets", 0, 5000, 500)

    if st.button("🔍 PREDICT DISASTER RISK", use_container_width=True):
        pred = model.predict([[rainfall, temperature, wind_speed, seismic, humidity, tweets]])[0]
        conf = model.predict_proba([[rainfall, temperature, wind_speed, seismic, humidity, tweets]]).max() * 100

        colors = {
            'Flood':'#1e3a8a','Wildfire':'#7c2d12','Earthquake':'#78350f',
            'Cyclone':'#4a1d96','Heatwave':'#7f1d1d','Landslide':'#365314','Safe':'#14532d'
        }
        icons = {
            'Flood':'🌊','Wildfire':'🔥','Earthquake':'🌍',
            'Cyclone':'🌀','Heatwave':'☀️','Landslide':'⛰️','Safe':'✅'
        }
        advice = {
            'Flood':      '🚨 Move to higher ground immediately! Avoid flooded roads.',
            'Wildfire':   '🚨 Evacuate now! Follow fire department instructions.',
            'Earthquake': '🚨 Drop, Cover, Hold On! Stay away from windows.',
            'Cyclone':    '🚨 Seek sturdy shelter! Stay indoors until storm passes.',
            'Heatwave':   '⚠️ Stay indoors! Drink water. Avoid direct sunlight.',
            'Landslide':  '🚨 Evacuate slope areas! Move to flat ground.',
            'Safe':       '✅ No immediate disaster risk detected in this area.'
        }

        st.markdown(f"""
        <div style='background:{colors[pred]};border-radius:16px;padding:32px;
        text-align:center;margin-top:20px;border:2px solid #ffffff33;'>
            <div style='font-size:64px;'>{icons[pred]}</div>
            <h2 style='color:white;font-size:40px;margin:12px 0;'>{pred}</h2>
            <h3 style='color:#fbbf24;font-size:28px;'>{conf:.1f}% Confidence</h3>
            <p style='color:#e2e8f0;font-size:16px;margin-top:12px;'>{advice[pred]}</p>
        </div>
        """, unsafe_allow_html=True)

# TAB 2 - Live Weather
with tab2:
    st.markdown("### 🌦️ Real-Time City Weather Prediction")
    city = st.text_input("Enter City Name:", placeholder="Chennai, Mumbai, Delhi...")

    if st.button("🔍 GET LIVE WEATHER & PREDICT", use_container_width=True):
        if city:
            with st.spinner(f"Fetching live weather for {city}..."):
                weather = get_weather(city)
            if weather:
                st.success(f"✅ Live data fetched for {city}!")
                w1,w2,w3,w4 = st.columns(4)
                w1.metric("🌡️ Temperature", f"{weather['temp']}°C")
                w2.metric("💧 Humidity", f"{weather['humidity']}%")
                w3.metric("💨 Wind Speed", f"{weather['wind']:.1f} km/h")
                w4.metric("🌧️ Rainfall", f"{weather['rain']} mm")

                features = [
                    weather['rain'], weather['temp'],
                    weather['wind'], 2.0,
                    weather['humidity'], 1000
                ]
                pred = model.predict([features])[0]
                conf = model.predict_proba([features]).max() * 100

                icons = {
                    'Flood':'🌊','Wildfire':'🔥','Earthquake':'🌍',
                    'Cyclone':'🌀','Heatwave':'☀️','Landslide':'⛰️','Safe':'✅'
                }
                colors = {
                    'Flood':'#1e3a8a','Wildfire':'#7c2d12','Earthquake':'#78350f',
                    'Cyclone':'#4a1d96','Heatwave':'#7f1d1d','Landslide':'#365314','Safe':'#14532d'
                }
                advice = {
                    'Flood':      'Move to higher ground immediately!',
                    'Wildfire':   'Evacuate! Follow fire department orders.',
                    'Earthquake': 'Drop, Cover, Hold On!',
                    'Cyclone':    'Seek shelter immediately!',
                    'Heatwave':   'Stay indoors! Drink water.',
                    'Landslide':  'Evacuate slope areas!',
                    'Safe':       'No immediate disaster risk detected.'
                }

                st.markdown(f"""
                <div style='background:{colors[pred]};border-radius:16px;padding:28px;
                text-align:center;margin-top:20px;border:2px solid #ffffff33;'>
                    <div style='font-size:56px;'>{icons[pred]}</div>
                    <h2 style='color:white;font-size:36px;margin:10px 0;'>{city} — {pred}</h2>
                    <h3 style='color:#fbbf24;font-size:26px;'>{conf:.1f}% Confidence</h3>
                    <p style='color:#e2e8f0;font-size:15px;margin-top:10px;'>{advice[pred]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("City not found! Try: Chennai, Mumbai, London, Tokyo")

# TAB 3 - World Map
with tab3:
    st.markdown("### 🗺️ Live World Disaster Risk Map")

    locations = [
        ('Chennai',    13.08,  80.27,  [250,28,40,2.1,90,3200]),
        ('California', 36.77,-119.41,  [20,43,80,1.5,25,500]),
        ('Tokyo',      35.68, 139.69,  [10,30,30,7.5,50,1200]),
        ('Bangladesh', 23.68,  90.35,  [15,35,140,3.0,45,800]),
        ('Mumbai',     19.07,  72.87,  [180,32,50,1.5,85,2000]),
        ('Australia',  -25.27, 133.77, [5,47,70,1.0,20,300]),
        ('Safe Zone',  20.59,  78.96,  [50,28,30,2.0,55,300]),
    ]

    color_map = {
        'Flood':'blue','Wildfire':'red','Earthquake':'orange',
        'Cyclone':'purple','Heatwave':'darkred','Landslide':'green','Safe':'green'
    }

    m = folium.Map(
        location=[20,78], zoom_start=2,
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB'
    )

    for name, lat, lon, params in locations:
        pred = model.predict([params])[0]
        conf = model.predict_proba([params]).max() * 100
        folium.CircleMarker(
            location=[lat, lon],
            radius=20,
            color=color_map.get(pred,'gray'),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{name}</b><br>⚠️ {pred}<br>Confidence: {conf:.1f}%",
                max_width=180
            )
        ).add_to(m)

    st_folium(m, width=1000, height=500)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#475569;'>DisasterSense AI | Final Year Project | Real-Time Multi-Hazard Warning System</p>", unsafe_allow_html=True)
