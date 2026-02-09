import os
import time
import requests
import streamlit as st
from src.serving.redis_queue import RedisClient

API_URL = os.getenv("API_URL", "http://localhost:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

st.set_page_config(page_title="Predictive Maintenance Monitor", layout="wide")
st.title("Predictive Maintenance Monitor (MVP)")

redis_client = RedisClient(REDIS_URL)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Manual prediction")
    default_feat = "0.1,0.2,0.15,0.18,0.12,0.09,0.11,0.2,0.3,0.25"
    feat_str = st.text_area("Features (comma-separated)", default_feat, height=100)

    if st.button("Predict"):
        features = [float(x.strip()) for x in feat_str.split(",") if x.strip()]
        r = requests.post(f"{API_URL}/predict-window", json={"features": features}, timeout=10)
        st.write(r.json())

        out = r.json()
        if out.get("severity") in ("MEDIUM", "HIGH"):
            requests.post(
                f"{API_URL}/event",
                json={
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": out["severity"],
                    "anomaly_score": out["anomaly_score"],
                    "message": "Anomalous window detected (manual input)"
                },
                timeout=10
            )
            st.info("Event logged to Redis (events list).")

with col2:
    st.subheader("Recent alerts (Redis)")
    refresh = st.toggle("Auto-refresh", value=True)
    limit = st.slider("How many events", 10, 200, 50)

    placeholder = st.empty()
    while refresh:
        events = redis_client.get_events(limit=limit)
        placeholder.dataframe(events, use_container_width=True)
        time.sleep(2)

    if not refresh:
        events = redis_client.get_events(limit=limit)
        st.dataframe(events, use_container_width=True)