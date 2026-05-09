# 🚢 Maritime Anomaly Detection Dashboard

> Real-time detection of suspicious cargo vessel behaviour using live AIS telemetry, unsupervised machine learning, and interactive visualisation.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [How It Works](#how-it-works)
3. [Machine Learning Models](#machine-learning-models)
4. [Architecture](#architecture)
5. [Installation & Usage](#installation--usage)
6. [Configuration](#configuration)
7. [Insights & Interpretation](#insights--interpretation)
8. [Limitations](#limitations)

---

## Business Problem

Illegal, Unreported, and Unregulated (IUU) fishing, dark-fleet tanker
bunkering, sanctions evasion, and ship-to-ship (STS) transfers in
restricted waters represent **multi-billion-dollar threats** to global
maritime security.  Vessels involved in these activities often exhibit
detectable behavioural patterns:

| Suspicious Behaviour | What It Looks Like in AIS Data |
|---|---|
| **AIS manipulation** | Gaps in transmission > 30 min |
| **Erratic speed / heading** | Cargo ship accelerating to 25+ knots or making sharp course changes |
| **Loitering / clustering** | Multiple vessels stationary in close proximity (potential STS) |
| **Dark-fleet activity** | Vessels operating in known restricted zones without proper declaration |

This dashboard ingests **live AIS data via WebSocket**, applies
statistical anomaly detection, and presents actionable intelligence on
an interactive global map.

---

## How It Works

```
+--------------+     +--------------+     +------------------+
|  WebSocket   |---->|  Data Clean  |---->|  ML Anomaly      |
|  AIS Stream  |     |  & Transform |     |  Detection       |
+--------------+     +--------------+     +--------+---------+
                                                   |
                                   +---------------+---------------+
                                   v               v               v
                           +--------------+ +------------+ +----------------+
                           |  Global Map  | |  Trend     | |  Distribution  |
                           |  (Plotly)    | |  Chart     | |  + Heatmap     |
                           +--------------+ +------------+ +----------------+
```

1. **Stream** — Connects to `wss://stream.aisstream.io` and collects raw AIS messages in real time.
2. **Clean** — Validates coordinates, filters unrealistic speeds (SOG > 30 kn for non-high-speed craft), imputes missing values, and engineers features.
3. **Detect** — Runs three complementary detectors:
   - **Isolation Forest** for behavioural anomalies
   - **DBSCAN** for spatial clustering
   - **AIS Gap Detector** for transmission silence
4. **Score** — Combines signals into a unified anomaly score (0-1) with severity buckets.
5. **Visualise** — Renders an interactive map and three analytical charts in a Streamlit dashboard.

---

## Machine Learning Models

### 1. Isolation Forest — Speed / Heading Anomalies

| Aspect | Detail |
|---|---|
| **Use case** | Detects vessels moving at unusual speeds or exhibiting erratic heading changes (e.g. a cargo ship behaving like a speedboat) |
| **Input features** | Speed Over Ground (SOG), Course Over Ground (COG), Rate of Turn (ROT) |
| **Why Isolation Forest?** | It is an unsupervised outlier-detection algorithm that isolates anomalies instead of profiling normal data. This is ideal because maritime anomaly patterns are diverse and not well-defined a priori. It scales linearly and handles mixed feature distributions well. |
| **Hyperparameters** | `contamination=0.05`, `n_estimators=100`, `random_state=42` |
| **Contribution to score** | 40% |

### 2. DBSCAN — Spatial Clustering

| Aspect | Detail |
|---|---|
| **Use case** | Identifies vessels loitering in restricted zones or forming unusual clusters (potential dark-fleet or STS activity) |
| **Input features** | Latitude, Longitude (haversine distance) |
| **Why DBSCAN?** | Unlike K-Means, DBSCAN does not require a pre-specified number of clusters and can detect arbitrary-shaped clusters and noise points. This is critical for maritime data where vessel groupings are irregular. |
| **Hyperparameters** | `eps=0.05` (~5 nautical miles), `min_samples=3` |
| **Contribution to score** | 35% |

### 3. AIS Gap Detector — Transmission Silence

| Aspect | Detail |
|---|---|
| **Use case** | Flags vessels with gaps > 30 minutes between AIS transmissions, a common indicator of intentional AIS manipulation |
| **Why rule-based?** | Gap detection is a well-defined temporal pattern that does not benefit from ML complexity; a straightforward rolling-window approach is more interpretable and computationally efficient. |
| **Contribution to score** | 25% |

### Combined Scoring

```
anomaly_score = 0.40 x IF_score_norm
              + 0.35 x DBSCAN_suspicious_flag
              + 0.25 x AIS_gap_flag

Severity buckets:
  Normal   : score < 0.20  (green)
  Low      : 0.20 <= score < 0.50  (yellow)
  Medium   : 0.50 <= score < 0.80  (orange)
  High     : score >= 0.80  (red)
```

---

## Architecture

```
Maritime_Anomaly_Detector/
├── app.py                 # Streamlit dashboard (entry point)
├── config.py              # Centralised configuration (dataclasses)
├── data_stream.py         # WebSocket AIS ingestion
├── data_cleaner.py        # Validation, filtering, feature engineering
├── anomaly_detector.py    # Isolation Forest, DBSCAN, Gap Detector, Scorer
├── visualizations.py      # Plotly map + charts
├── sample_data.py         # Realistic demo data with seeded anomalies
├── utils.py               # Shared helpers
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment variables
└── README.md              # This file
```

Each module is **single-responsibility**, **PEP 8 compliant**, and importable for unit testing.

---

## Installation & Usage

### Prerequisites

- Python >= 3.9
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Maritime_Anomaly_Detector.git
cd Maritime_Anomaly_Detector

# 2. Create a virtual environment (recommended)
python3 -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Activate on Windows (Command Prompt):
.venv\Scripts\activate.bat

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key (see Configuration section below)
cp .env.example .env
# Edit .env and add your AISStream API key

# 5. Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Controls

| Control | Description |
|---|---|
| **Data Source** | Choose between "Live AIS Stream" (requires API key) or "Sample Data (Demo)" for offline exploration. |
| **Messages to collect** | Slider (100-5000). More messages = richer data but longer load time. |
| **Refresh Data** | Re-fetches AIS data and re-runs the full pipeline. |

---

## Configuration

### API Key Setup (Required for Live Data)

This project uses an AISStream API key to access live maritime data. **Never commit your API key to version control.**

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your key:
   ```text
   AIS_API_KEY=your_actual_api_key_here
   ```

3. The `.env` file is listed in `.gitignore` and will never be uploaded to GitHub.

> **Note:** If no API key is configured, the dashboard will automatically fall back to sample data mode so you can still explore all visualizations and ML models.

### Model Tuning

All settings live in `config.py` as nested `@dataclass` structures:

```python
from config import AppConfig

cfg = AppConfig()
cfg.isolation_forest.contamination = 0.10   # expect more anomalies
cfg.dbscan.eps = 0.03                       # tighter spatial clusters
cfg.ais_gap_threshold_minutes = 15          # more sensitive gap detection
```

---

## Insights & Interpretation

### What the Dashboard Tells You

| Visualisation | Business Insight |
|---|---|
| **Global Anomaly Map** | See exactly **where** suspicious activity is happening. Red markers indicate high-confidence anomalies. Click any vessel for its IMO, MMSI, speed, and flagged reason. |
| **Trend Chart** | Tracks anomaly severity counts over time. Spikes may correlate with known events (port closures, sanctions enforcement windows). |
| **Distribution Chart** | Shows the spread of anomaly scores. A heavy right-tail suggests systemic anomalous behaviour in the observed fleet. |
| **Correlation Heatmap** | Reveals which navigation features (SOG, ROT, heading divergence) are most correlated with the anomaly score, guiding further investigation. |
| **Top Flagged Vessels Table** | A ranked shortlist for human analysts to investigate, with IMO/MMSI for cross-referencing against sanctions lists. |

### Example Investigation Workflow

1. A vessel appears **red** on the map with reason *"Speed/Heading Anomaly, Suspicious Cluster (STS?)"*.
2. Analyst clicks the marker -> sees IMO `9876543`, MMSI `123456789`.
3. Cross-references IMO against UN sanctions list -> match found.
4. Escalates to maritime intelligence team for further action.

---

## Limitations

| Limitation | Mitigation |
|---|---|
| Snapshot data - the WebSocket provides a time-bounded sample, not a full historical track. | Increase `max_messages` or run repeated collections. |
| Unsupervised models produce false positives. | Human-in-the-loop review via the Top Flagged table. |
| AIS data can be spoofed or manipulated. | Gap detector partially addresses this; multi-source validation (SAR, VMS) is recommended for operational deployments. |
| DBSCAN `eps` is distance-based and may miss clusters at different scales. | Tune `eps` per region or switch to HDBSCAN for hierarchical clustering. |

---

## License

This project is provided for educational and analytical purposes.

---

**Built with** Streamlit, scikit-learn, Plotly, websockets, pandas
