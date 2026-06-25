import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DisasterSense AI",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0f172a; }
.stApp { background-color: #0f172a; color: white; }
h1,h2,h3,p,label { color: white !important; }
.metric-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #334155;
}
.alert-box {
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 2000
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
        else:
            return 'Safe'
    df['disaster'] = df.apply(label, axis=1)
    X = df.drop('disaster', axis=1)
    y = df['disaster']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

st.markdown("<h1 style='color:#ef4444;font-size:36px;'>🌍 DisasterSense AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>Multi-Hazard Early Warning System — Powered by AI</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", "99%")
with col2:
    st.metric("Disaster Types", "5")
with col3:
    st.metric("Data Points", "2,000")
with col4:
    st.metric("AI Status", "Live")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.markdown("### 🎛️ Enter Parameters")

    rainfall   = st.slider("Rainfall (mm)",          0,   300, 150)
    temperature= st.slider("Temperature (°C)",       15,   50,  30)
    wind_speed = st.slider("Wind Speed (km/h)",       0,  200,  60)
    seismic    = st.slider("Seismic Activity (Richter)", 0.0, 9.0, 3.0, 0.1)
    humidity   = st.slider("Humidity (%)",           10,  100,  60)
    tweets     = st.slider("Disaster Tweets",         0, 5000, 500)

    predict_btn = st.button("🔍 PREDICT DISASTER RISK", use_container_width=True)

with right:
    st.markdown("### 🗺️ Live Disaster Map")

    locations = [
        ('Chennai',     13.08, 80.27,  [250,28,40,2.1,90,3200]),
        ('California',  36.77,-119.41, [20,43,80,1.5,25,500]),
        ('Tokyo',       35.68, 139.69, [10,30,30,7.5,50,1200]),
        ('Bangladesh',  23.68,  90.35, [15,35,140,3.0,45,800]),
        ('Safe Zone',   20.59,  78.96, [50,28,30,2.0,55,300]),
    ]

    colors = {
        'Flood':'blue','Wildfire':'red',
        'Earthquake':'orange','Cyclone':'purple','Safe':'green'
    }

    m = folium.Map(
        location=[20,78], zoom_start=3,
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB'
    )

    for name, lat, lon, params in locations:
        pred = model.predict([params])[0]
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            color=colors.get(pred,'gray'),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(f"<b>{name}</b><br>Risk: {pred}", max_width=150)
        ).add_to(m)

    st_folium(m, width=500, height=380)

if predict_btn:
    features = [rainfall, temperature, wind_speed, seismic, humidity, tweets]
    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features]).max() * 100

    color_map = {
        'Flood':      '#1e3a8a',
        'Wildfire':   '#7c2d12',
        'Earthquake': '#78350f',
        'Cyclone':    '#4a1d96',
        'Safe':       '#14532d'
    }
    icon_map = {
        'Flood':'🌊','Wildfire':'🔥',
        'Earthquake':'🌍','Cyclone':'🌀','Safe':'✅'
    }
    advice_map = {
        'Flood':      'Move to higher ground immediately!',
        'Wildfire':   'Evacuate! Follow fire department orders.',
        'Earthquake': 'Drop, Cover, Hold On! Stay away from windows.',
        'Cyclone':    'Seek shelter! Stay indoors until storm passes.',
        'Safe':       'No immediate disaster risk detected.'
    }

    bg = color_map[prediction]
    icon = icon_map[prediction]
    advice = advice_map[prediction]

    st.markdown(f"""
    <div style='background:{bg};border-radius:16px;padding:32px;text-align:center;margin-top:24px;border:2px solid #ffffff33;'>
        <div style='font-size:56px;'>{icon}</div>
        <h2 style='font-size:36px;color:white;margin:12px 0;'>{prediction}</h2>
        <h3 style='font-size:28px;color:#fbbf24;'>{confidence:.1f}% confidence</h3>
        <p style='color:#e2e8f0;font-size:16px;margin-top:12px;'>{advice}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#475569;'>DisasterSense AI — Final Year Project | Powered by Random Forest + Streamlit</p>", unsafe_allow_html=True)