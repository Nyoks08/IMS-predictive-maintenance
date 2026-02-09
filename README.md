# IMS-predictive-maintenance
Three (3) data sets are included in the data packet (IMS-Rexnord Bearing Data.zip). Each data set describes a test-to-failure experiment.

# Real-time Predictive Maintenance (MVP)

This project simulates real-time industrial sensor monitoring and detects anomalies using a production-style pipeline:
- FastAPI inference service
- Sliding-window features (MVP uses manual features; dataset integration is next)
- Event logging with Redis
- Streamlit monitoring dashboard

## Quickstart (local)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Start Redis (Option A: Docker)
docker run -p 6379:6379 redis:7-alpine

# Start API
uvicorn api.main:app --reload --port 8000

# Start UI (new terminal)
streamlit run ui/app.py --server.port 8501
94ab53d (Initial IMS predictive maintenance pipeline (training, API, Redis workers))
