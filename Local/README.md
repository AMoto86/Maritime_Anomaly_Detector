# 🚢 Maritime Anomaly Detection Dashboard - Local Deployment

This is the **local deployment** version of the Maritime Anomaly Detector.
It includes your API key for immediate use on this machine.

## ⚠️ Security Warning
- This folder contains your API key in `config.py` and `.env`.
- **DO NOT** commit this folder to any public repository.
- The parent directory contains the secure, public-ready version.

## Quick Start

```bash
# 1. Navigate to this Local folder
cd Local

# 2. Create a virtual environment (if not already done)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## Data Source Options
- **Live AIS Stream**: Uses your API key to fetch real-time maritime data.
- **Sample Data (Demo)**: Uses realistic simulated data for testing without an API key.

## Troubleshooting
- If the live stream fails, check that your API key is active at [aisstream.io](https://aisstream.io).
- Switch to "Sample Data (Demo)" in the sidebar to verify the ML pipeline works.
