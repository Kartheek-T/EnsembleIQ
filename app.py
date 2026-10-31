import os
from datetime import datetime
 
import numpy as np
import pandas as pd
import streamlit as st
 
try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
 
 
# ============================================================
# CONFIG / CONSTANTS
# ============================================================
 
APP_TITLE = "EnsembleIQ"
APP_TAGLINE = "AI Fraud Intelligence"
 
DATA_DIR = "data"
MODEL_DIR = "models"
DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
 
# Feature order preserved exactly as it appears in the original dataset.
# Time and Amount are scaled in-place using scaler.pkl; V1-V28 are used
# as-is, since that is how the saved models were trained.
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time"] + V_FEATURES + ["Amount"]
 
# Registry of all saved models this dashboard knows how to load.
# key -> (display name, filename, strategy/category)
MODEL_REGISTRY = {
    "random_forest": {
        "label": "Random Forest",
        "file": "random_forest.pkl",
        "category": "Bagging",
    },
    "extra_trees": {
        "label": "Extra Trees",
        "file": "extra_trees.pkl",
        "category": "Bagging",
    },
    "adaboost": {
        "label": "AdaBoost",
        "file": "adaboost.pkl",
        "category": "Boosting",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "file": "gradient_boosting.pkl",
        "category": "Boosting",
    },
    "xgboost": {
        "label": "XGBoost",
        "file": "xgboost.pkl",
        "category": "Boosting",
    },
    "lightgbm": {
        "label": "LightGBM",
        "file": "lightgbm.pkl",
        "category": "Boosting",
    },
    "catboost": {
        "label": "CatBoost",
        "file": "catboost.pkl",
        "category": "Boosting",
    },
}
 
PRIMARY_MODEL_KEY = "extra_trees"
MAX_DEMO_ROWS = 5000  # cap how many rows we pull into memory for demo browsing
 
 
# ============================================================
# HTML RENDER HELPER
# ============================================================
# IMPORTANT: st.markdown() runs its input through a Markdown parser even
# when unsafe_allow_html=True. Markdown treats any line that starts with
# 4+ spaces of indentation as a *fenced/indented code block*, which
# escapes the HTML instead of rendering it - this was the root cause of
# raw <div> markup appearing as literal text on the page. Every custom
# HTML string in this file is passed through render_html(), which strips
# leading whitespace from every line before handing it to st.markdown so
# Markdown never mistakes our HTML for a code block.
 
def render_html(html_string: str) -> None:
    """Render a block of raw HTML safely inside Streamlit."""
    cleaned_lines = [line.lstrip() for line in html_string.strip("\n").split("\n")]
    cleaned = "\n".join(cleaned_lines)
    st.markdown(cleaned, unsafe_allow_html=True)
 
 
# ============================================================
# PAGE CONFIG
# ============================================================
 
st.set_page_config(
    page_title=f"{APP_TITLE} | {APP_TAGLINE}",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
# ------------------------------------------------------------
# GLOBAL THEME / DESIGN SYSTEM
# ------------------------------------------------------------
# Token system:
#   bg base        #060A14   near-black navy
#   bg surface      #0D1424   card navy
#   bg surface-2    #121B32   elevated glass panel
#   border subtle   rgba(140,170,255,0.14)
#   accent cyan     #22D3EE
#   accent blue     #3B82F6
#   accent purple   #8B5CF6
#   accent danger   #FB5B7B
#   accent success  #34E0A1
#   text primary    #E9EDF9
#   text muted      #8792AC
# Display face: Space Grotesk / Body: Inter / Data-mono: JetBrains Mono
render_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
 
:root {
--eiq-bg: #060A14;
--eiq-surface: #0D1424;
--eiq-surface-2: #121B32;
--eiq-surface-3: #0A1120;
--eiq-border: rgba(140,170,255,0.14);
--eiq-border-strong: rgba(140,170,255,0.30);
--eiq-cyan: #22D3EE;
--eiq-blue: #3B82F6;
--eiq-purple: #8B5CF6;
--eiq-danger: #FB5B7B;
--eiq-success: #34E0A1;
--eiq-text: #E9EDF9;
--eiq-muted: #8792AC;
--eiq-grad-brand: linear-gradient(135deg, var(--eiq-cyan) 0%, var(--eiq-blue) 55%, var(--eiq-purple) 100%);
}
 
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
 
.stApp {
background:
radial-gradient(circle at 12% -10%, rgba(139,92,246,0.16), transparent 42%),
radial-gradient(circle at 88% 0%, rgba(34,211,238,0.14), transparent 40%),
var(--eiq-bg);
color: var(--eiq-text);
}
 
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 620px; }
 
h1, h2, h3, h4, .eiq-display { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
 
.eiq-mono { font-family: 'JetBrains Mono', monospace; }
 
/* ---------- Header / brand ---------- */
.eiq-brand-row { display: flex; align-items: center; gap: 12px; }
.eiq-logo-mark {
width: 42px; height: 42px; border-radius: 12px;
background: var(--eiq-grad-brand);
display: flex; align-items: center; justify-content: center;
font-size: 19px; box-shadow: 0 6px 18px rgba(59,130,246,0.35);
position: relative; flex-shrink: 0;
}
.eiq-logo-mark::after {
content: "";
position: absolute; inset: 2px;
border-radius: 10px;
border: 1px solid rgba(255,255,255,0.35);
}
.eiq-brand-title {
font-size: 22px; font-weight: 700; margin: 0; line-height: 1.1;
background: var(--eiq-grad-brand);
-webkit-background-clip: text; background-clip: text; color: transparent;
}
.eiq-brand-sub {
font-family: 'JetBrains Mono', monospace;
font-size: 10.5px; letter-spacing: 0.14em; text-transform: uppercase;
color: var(--eiq-muted); margin-top: 1px;
}
 
.eiq-status-pill {
display: inline-flex; align-items: center; gap: 7px;
padding: 6px 13px; border-radius: 999px;
border: 1px solid var(--eiq-border-strong);
background: rgba(18,27,50,0.7); backdrop-filter: blur(6px);
font-family: 'JetBrains Mono', monospace;
font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em;
color: var(--eiq-text);
}
.eiq-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--eiq-success);
box-shadow: 0 0 8px var(--eiq-success); animation: eiq-pulse 1.7s infinite; flex-shrink: 0; }
.eiq-dot.warn { background: var(--eiq-danger); box-shadow: 0 0 8px var(--eiq-danger); }
.eiq-sub-status { font-size: 10.5px; color: var(--eiq-muted); text-align: right; margin-top: 5px; }
 
/* ---------- Glass surfaces ---------- */
.eiq-glass {
background: linear-gradient(180deg, var(--eiq-surface-2) 0%, var(--eiq-surface) 100%);
border: 1px solid var(--eiq-border);
border-radius: 18px;
padding: 22px 22px;
margin-bottom: 14px;
box-shadow: 0 10px 30px rgba(0,0,0,0.35);
animation: eiq-fade-in 0.4s ease-out;
}
.eiq-glass-flat {
background: var(--eiq-surface);
border: 1px solid var(--eiq-border);
border-radius: 14px;
padding: 16px 18px;
margin-bottom: 10px;
}
 
/* ---------- Flow / pipeline stepper ---------- */
.eiq-flow { display: flex; align-items: center; justify-content: space-between;
margin: 4px 0 18px 0; padding: 0 2px; }
.eiq-flow-step { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: 1; position: relative; }
.eiq-flow-node {
width: 26px; height: 26px; border-radius: 50%;
border: 1.5px solid var(--eiq-border-strong);
background: var(--eiq-surface-3);
display: flex; align-items: center; justify-content: center;
font-size: 11px; color: var(--eiq-muted); font-weight: 700;
transition: all 0.35s ease; z-index: 1;
}
.eiq-flow-node.active {
background: var(--eiq-grad-brand); color: #06101F; border-color: transparent;
box-shadow: 0 0 14px rgba(34,211,238,0.55);
}
.eiq-flow-label { font-size: 9px; letter-spacing: 0.05em; text-transform: uppercase;
color: var(--eiq-muted); font-family: 'JetBrains Mono', monospace; text-align: center; }
.eiq-flow-label.active { color: var(--eiq-cyan); }
.eiq-flow-line { position: absolute; top: 13px; left: 50%; width: 100%; height: 1.5px;
background: var(--eiq-border); z-index: 0; }
.eiq-flow-line.active { background: var(--eiq-grad-brand); }
.eiq-flow-step:last-child .eiq-flow-line { display: none; }
 
/* ---------- Credit card visual ---------- */
.eiq-card-visual {
position: relative; overflow: hidden;
border-radius: 20px; padding: 24px 24px 20px 24px;
min-height: 178px;
background:
radial-gradient(circle at 100% 0%, rgba(139,92,246,0.55), transparent 55%),
radial-gradient(circle at 0% 100%, rgba(34,211,238,0.35), transparent 55%),
linear-gradient(135deg, #0E1730 0%, #131C36 45%, #0A1120 100%);
border: 1px solid rgba(140,170,255,0.22);
box-shadow: 0 18px 40px rgba(4,8,20,0.55), inset 0 1px 0 rgba(255,255,255,0.05);
animation: eiq-fade-in 0.5s ease-out;
margin-bottom: 16px;
}
.eiq-card-visual::before {
content: ""; position: absolute; top: -70%; left: -25%;
width: 55%; height: 240%;
background: linear-gradient(120deg, transparent, rgba(255,255,255,0.10), transparent);
transform: rotate(18deg);
animation: eiq-sheen 4.5s ease-in-out infinite;
}
.eiq-card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px; position: relative; z-index: 1; }
.eiq-card-chip {
width: 40px; height: 30px; border-radius: 7px;
background: linear-gradient(135deg, #E4C97A, #B98F3D);
position: relative; box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
.eiq-card-chip::before, .eiq-card-chip::after {
content: ""; position: absolute; left: 6px; right: 6px; height: 1px; background: rgba(0,0,0,0.35);
}
.eiq-card-chip::before { top: 10px; } .eiq-card-chip::after { top: 19px; }
.eiq-card-nfc { font-size: 17px; color: rgba(233,237,249,0.55); transform: rotate(90deg); }
.eiq-card-number {
font-family: 'JetBrains Mono', monospace;
font-size: 19px; letter-spacing: 0.14em; font-weight: 600;
color: var(--eiq-text); margin-bottom: 20px; position: relative; z-index: 1;
text-shadow: 0 0 18px rgba(34,211,238,0.25);
}
.eiq-card-foot { display: flex; justify-content: space-between; align-items: flex-end; position: relative; z-index: 1; }
.eiq-card-foot-block .eiq-card-label {
font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 0.12em;
text-transform: uppercase; color: rgba(233,237,249,0.45); margin-bottom: 3px;
}
.eiq-card-foot-block .eiq-card-value { font-size: 15px; font-weight: 600; color: var(--eiq-text); }
.eiq-card-brand {
font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 13px;
background: var(--eiq-grad-brand); -webkit-background-clip: text; background-clip: text; color: transparent;
letter-spacing: 0.02em;
}
.eiq-scan-line {
position: absolute; left: 0; right: 0; height: 2px;
background: linear-gradient(90deg, transparent, var(--eiq-cyan), transparent);
box-shadow: 0 0 12px var(--eiq-cyan);
animation: eiq-scan-sweep 1.4s linear infinite;
z-index: 2;
}
 
/* ---------- Buttons ---------- */
div.stButton > button {
width: 100%; border-radius: 12px; padding: 0.7em 1em;
font-family: 'Inter', sans-serif; font-weight: 600; font-size: 14.5px;
border: 1px solid var(--eiq-border-strong);
background: var(--eiq-surface-2); color: var(--eiq-text);
transition: all 0.18s ease;
}
div.stButton > button:hover {
border-color: var(--eiq-cyan); color: var(--eiq-cyan);
box-shadow: 0 0 0 1px var(--eiq-cyan), 0 8px 20px rgba(34,211,238,0.15);
transform: translateY(-1px);
}
div.stButton > button:active { transform: translateY(0) scale(0.98); }
div.stButton > button[kind="primary"] {
background: var(--eiq-grad-brand); color: #06101F; border: none;
box-shadow: 0 8px 22px rgba(59,130,246,0.35);
}
div.stButton > button[kind="primary"]:hover {
box-shadow: 0 10px 28px rgba(139,92,246,0.5); color: #06101F;
filter: brightness(1.07);
}
 
/* ---------- Result / risk visuals ---------- */
.eiq-result-headline { font-family: 'Space Grotesk', sans-serif; font-size: 25px; font-weight: 700;
margin-bottom: 3px; animation: eiq-pop-in 0.35s ease-out; }
.eiq-result-headline.fraud { color: var(--eiq-danger); text-shadow: 0 0 22px rgba(251,91,123,0.35); }
.eiq-result-headline.clear { color: var(--eiq-success); text-shadow: 0 0 22px rgba(52,224,161,0.3); }
.eiq-prob-track { width: 100%; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.06);
overflow: hidden; margin: 8px 0 4px 0; border: 1px solid var(--eiq-border); }
.eiq-prob-fill { height: 100%; border-radius: 999px; animation: eiq-grow-width 0.8s ease-out; }
.eiq-prob-fill.fraud { background: linear-gradient(90deg, #FB5B7B, #F59E0B); }
.eiq-prob-fill.clear { background: linear-gradient(90deg, #22D3EE, #34E0A1); }
.eiq-risk-badge { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 11px;
font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; border: 1px solid; }
.eiq-risk-badge.low { color: var(--eiq-success); border-color: var(--eiq-success); background: rgba(52,224,161,0.08); }
.eiq-risk-badge.medium { color: #F5C542; border-color: #F5C542; background: rgba(245,197,66,0.08); }
.eiq-risk-badge.high { color: #F59E0B; border-color: #F59E0B; background: rgba(245,158,11,0.1); }
.eiq-risk-badge.critical { color: var(--eiq-danger); border-color: var(--eiq-danger); background: rgba(251,91,123,0.1); }
 
/* ---------- Model rows ---------- */
.eiq-section-title { font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 700;
color: var(--eiq-text); margin: 4px 0 10px 0; display: flex; align-items: center; gap: 8px; }
.eiq-section-title .eiq-tag { font-family: 'JetBrains Mono', monospace; font-size: 10px;
color: var(--eiq-muted); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }
.eiq-model-row {
border-radius: 12px; padding: 13px 16px; margin-bottom: 8px;
background: var(--eiq-surface); border: 1px solid var(--eiq-border);
transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
animation: eiq-slide-up 0.3s ease-out;
}
.eiq-model-row:hover { transform: translateY(-1px); border-color: var(--eiq-border-strong);
box-shadow: 0 6px 16px rgba(0,0,0,0.3); }
.eiq-model-name { font-weight: 600; font-size: 14px; color: var(--eiq-text); }
.eiq-model-cat { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--eiq-muted);
letter-spacing: 0.05em; text-transform: uppercase; margin-top: 1px; }
.eiq-badge { display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: 10.5px;
font-weight: 700; letter-spacing: 0.04em; border: 1px solid; }
.eiq-badge.fraud { color: var(--eiq-danger); border-color: var(--eiq-danger); background: rgba(251,91,123,0.1); }
.eiq-badge.clear { color: var(--eiq-success); border-color: var(--eiq-success); background: rgba(52,224,161,0.08); }
.eiq-mini-track { width: 100%; height: 5px; border-radius: 999px; background: rgba(255,255,255,0.07);
overflow: hidden; margin-top: 8px; }
.eiq-mini-fill { height: 100%; border-radius: 999px; }
 
.eiq-small-label { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; text-transform: uppercase;
letter-spacing: 0.09em; color: var(--eiq-muted); margin-bottom: 3px; }
 
/* ---------- Consensus ---------- */
.eiq-consensus-number { font-family: 'Space Grotesk', sans-serif; font-size: 21px; font-weight: 700;
color: var(--eiq-text); margin-bottom: 2px; }
.eiq-consensus-msg { font-size: 13px; color: var(--eiq-muted); margin-bottom: 10px; }
 
/* ---------- Streamlit widget overrides ---------- */
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
background: var(--eiq-surface-2) !important; color: var(--eiq-text) !important;
border: 1px solid var(--eiq-border-strong) !important; border-radius: 10px !important;
font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stNumberInput"] button { background: var(--eiq-surface-3) !important; border-color: var(--eiq-border) !important; }
div[data-testid="stFileUploaderDropzone"] {
background: var(--eiq-surface) !important; border: 1.5px dashed var(--eiq-border-strong) !important;
border-radius: 14px !important;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--eiq-cyan) !important; }
div[role="radiogroup"] { gap: 4px; }
div[data-testid="stMetric"] {
background: var(--eiq-surface); border: 1px solid var(--eiq-border); border-radius: 14px;
padding: 12px 14px 8px 14px;
}
div[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: 10.5px !important;
text-transform: uppercase; letter-spacing: 0.06em; color: var(--eiq-muted) !important; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--eiq-text) !important; }
.streamlit-expanderHeader, div[data-testid="stExpander"] {
background: var(--eiq-surface) !important; border: 1px solid var(--eiq-border) !important;
border-radius: 12px !important; color: var(--eiq-text) !important;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--eiq-border); border-radius: 12px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 12px; }
.stDownloadButton button {
border-radius: 12px !important; border: 1px solid var(--eiq-border-strong) !important;
background: var(--eiq-surface-2) !important; color: var(--eiq-text) !important; font-weight: 600 !important;
}
.stDownloadButton button:hover { border-color: var(--eiq-cyan) !important; color: var(--eiq-cyan) !important; }
hr { border-color: var(--eiq-border) !important; }
::-webkit-scrollbar { height: 8px; width: 8px; }
::-webkit-scrollbar-thumb { background: var(--eiq-border-strong); border-radius: 8px; }
 
@keyframes eiq-fade-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes eiq-slide-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes eiq-pop-in { 0% { opacity: 0; transform: scale(0.96); } 100% { opacity: 1; transform: scale(1); } }
@keyframes eiq-pulse { 0% { opacity: 1; } 50% { opacity: 0.35; } 100% { opacity: 1; } }
@keyframes eiq-sheen { 0% { transform: translateX(-50%) rotate(18deg); } 100% { transform: translateX(230%) rotate(18deg); } }
@keyframes eiq-grow-width { from { width: 0%; } }
@keyframes eiq-scan-sweep { 0% { top: 4%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 96%; opacity: 0; } }
 
@media (max-width: 640px) {
.eiq-brand-title { font-size: 19px; }
.eiq-card-number { font-size: 16px; letter-spacing: 0.10em; }
.eiq-flow-label { font-size: 8px; }
}
</style>
"""
)
 
 
# ============================================================
# CACHED LOADERS
# ============================================================
 
@st.cache_resource(show_spinner=False)
def load_models():
    """Load every model file listed in MODEL_REGISTRY. Returns a dict of
    key -> loaded model object (or None if that specific file failed to
    load), plus a status dict of key -> error message (empty if ok)."""
    models = {}
    errors = {}
 
    if joblib is None:
        for key in MODEL_REGISTRY:
            models[key] = None
            errors[key] = "joblib is not installed in this environment."
        return models, errors
 
    for key, meta in MODEL_REGISTRY.items():
        path = os.path.join(MODEL_DIR, meta["file"])
        if not os.path.exists(path):
            models[key] = None
            errors[key] = f"File not found: {path}"
            continue
        try:
            models[key] = joblib.load(path)
            errors[key] = ""
        except Exception as exc:  # noqa: BLE001
            models[key] = None
            errors[key] = f"Failed to load: {exc}"
 
    return models, errors
 
 
@st.cache_resource(show_spinner=False)
def load_scaler():
    if joblib is None:
        return None, "joblib is not installed in this environment."
    if not os.path.exists(SCALER_PATH):
        return None, f"Scaler file not found: {SCALER_PATH}"
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to load scaler: {exc}"
 
 
@st.cache_data(show_spinner=False)
def load_demo_dataset():
    """Load a capped number of rows from the dataset for demo browsing,
    so we don't repeatedly read the full ~284,807-row file."""
    if not os.path.exists(DATA_PATH):
        return None, f"Dataset not found: {DATA_PATH}"
    try:
        df = pd.read_csv(DATA_PATH, nrows=MAX_DEMO_ROWS)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return None, f"Dataset is missing required columns: {', '.join(missing)}"
        return df, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to read dataset: {exc}"
 
 
# ============================================================
# HELPERS (ML logic - unchanged)
# ============================================================
 
def scale_time_amount(time_vals, amount_vals, scaler):
    """Apply the saved scaler to Time/Amount pairs. Accepts array-like
    inputs and returns (scaled_time, scaled_amount) arrays."""
    raw = np.column_stack([np.asarray(time_vals, dtype=float),
                            np.asarray(amount_vals, dtype=float)])
    scaled = scaler.transform(raw)
    return scaled[:, 0], scaled[:, 1]
 
 
def build_feature_frame(row, scaler):
    """Build a single-row DataFrame in FEATURE_COLUMNS order, with Time
    and Amount scaled using scaler.pkl and V1-V28 left untouched."""
    scaled_time, scaled_amount = scale_time_amount([row["Time"]], [row["Amount"]], scaler)
    values = {}
    for col in FEATURE_COLUMNS:
        if col == "Time":
            values[col] = scaled_time[0]
        elif col == "Amount":
            values[col] = scaled_amount[0]
        else:
            values[col] = row[col]
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)
 
 
def build_feature_frame_batch(df, scaler):
    """Vectorized version of build_feature_frame for a whole DataFrame."""
    scaled_time, scaled_amount = scale_time_amount(df["Time"].values, df["Amount"].values, scaler)
    out = df[FEATURE_COLUMNS].copy()
    out["Time"] = scaled_time
    out["Amount"] = scaled_amount
    return out
 
 
def fraud_probability(model, X):
    """Return (prediction, fraud_probability) using the model's own
    predict / predict_proba. Never fabricated."""
    pred = model.predict(X)[0]
    proba_row = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        fraud_idx = classes.index(1)
    else:
        fraud_idx = len(proba_row) - 1
    prob_fraud = float(proba_row[fraud_idx])
    return int(pred), prob_fraud
 
 
def fraud_probability_batch(model, X):
    preds = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    fraud_idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    probs = proba[:, fraud_idx]
    return preds.astype(int), probs
 
 
def risk_level(prob):
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"
 
 
def reset_transaction_state():
    st.session_state.transaction_checked = False
    st.session_state.show_full_analysis = False
    st.session_state.primary_result = None
    st.session_state.all_model_results = None
 
 
def init_session_state():
    defaults = {
        "selected_transaction": 0,
        "transaction_checked": False,
        "show_full_analysis": False,
        "primary_result": None,
        "all_model_results": None,
        "input_mode": "Demo Transaction",
        "csv_results_df": None,
        "csv_selected_row": 0,
        "csv_show_row_analysis": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
 
# ============================================================
# UI COMPONENTS
# ============================================================
 
def render_header(models_ok_count, models_total, dataset_ok):
    all_ok = (models_ok_count == models_total) and dataset_ok
    status_text = "SYSTEM ONLINE" if all_ok else "SYSTEM DEGRADED"
    dot_class = "" if all_ok else "warn"
 
    left, right = st.columns([3, 2])
    with left:
        render_html(f"""
        <div class="eiq-brand-row">
            <div class="eiq-logo-mark">🛡️</div>
            <div>
                <p class="eiq-brand-title">{APP_TITLE}</p>
                <div class="eiq-brand-sub">{APP_TAGLINE}</div>
            </div>
        </div>
        """)
    with right:
        render_html(f"""
        <div style="text-align:right; margin-top:8px;">
            <span class="eiq-status-pill">
                <span class="eiq-dot {dot_class}"></span>{status_text}
            </span>
            <div class="eiq-sub-status">{models_ok_count}/{models_total} models loaded</div>
        </div>
        """)
    render_html("<div style='height:14px'></div>")
 
 
def render_flow(stage):
    """stage: one of 'card', 'check', 'scan', 'result', 'full'"""
    order = ["card", "check", "scan", "result", "full"]
    labels = ["Card", "Verify", "AI Scan", "Risk Result", "Full Analysis"]
    icons = ["💳", "🔎", "⚡", "📊", "🧩"]
    current_idx = order.index(stage)
 
    steps_html = ""
    for i, (key, label, icon) in enumerate(zip(order, labels, icons)):
        active = i <= current_idx
        node_cls = "active" if active else ""
        label_cls = "active" if active else ""
        line_cls = "active" if i < current_idx else ""
        steps_html += f"""
        <div class="eiq-flow-step">
            <div class="eiq-flow-line {line_cls}"></div>
            <div class="eiq-flow-node {node_cls}">{icon}</div>
            <div class="eiq-flow-label {label_cls}">{label}</div>
        </div>
        """
 
    render_html(f'<div class="eiq-flow">{steps_html}</div>')
 
 
def render_transaction_card(txn_ref, amount, source, scanning=False):
    scan_html = '<div class="eiq-scan-line"></div>' if scanning else ""
    render_html(f"""
    <div class="eiq-card-visual">
        {scan_html}
        <div class="eiq-card-top">
            <div class="eiq-card-chip"></div>
            <div class="eiq-card-nfc">📶</div>
        </div>
        <div class="eiq-card-number">TXN •••• •••• {txn_ref:06d}</div>
        <div class="eiq-card-foot">
            <div class="eiq-card-foot-block">
                <div class="eiq-card-label">Amount</div>
                <div class="eiq-card-value">${amount:,.2f}</div>
            </div>
            <div class="eiq-card-foot-block" style="text-align:right;">
                <div class="eiq-card-label">Source</div>
                <div class="eiq-card-value" style="font-size:12.5px;">{source}</div>
            </div>
            <div class="eiq-card-brand">EnsembleIQ</div>
        </div>
    </div>
    """)
 
 
def render_primary_result(pred, prob, model_label):
    level = risk_level(prob)
    verdict = "FRAUD DETECTED" if pred == 1 else "TRANSACTION CLEAR"
    state = "fraud" if pred == 1 else "clear"
    pct = prob * 100
 
    render_html(f"""
    <div class="eiq-glass">
        <div class="eiq-small-label">AI Analysis Result</div>
        <div class="eiq-result-headline {state}">{verdict}</div>
        <div style="font-size:13px; color:var(--eiq-muted); margin-bottom:12px;">
            Primary model: <b style="color:var(--eiq-text);">{model_label}</b>
            &nbsp;•&nbsp; Risk level:
            <span class="eiq-risk-badge {level.lower()}">{level}</span>
        </div>
        <div class="eiq-small-label">Fraud probability</div>
        <div class="eiq-prob-track">
            <div class="eiq-prob-fill {state}" style="width:{pct:.1f}%;"></div>
        </div>
        <div class="eiq-mono" style="font-size:13px; color:var(--eiq-text);">{pct:.2f}%</div>
    </div>
    """)
 
 
def render_model_row(label, category, pred, prob):
    verdict = "FRAUD" if pred == 1 else "CLEAR"
    state = "fraud" if pred == 1 else "clear"
    pct = prob * 100
    fill_color = "linear-gradient(90deg, #FB5B7B, #F59E0B)" if pred == 1 else "linear-gradient(90deg, #22D3EE, #34E0A1)"
    render_html(f"""
    <div class="eiq-model-row">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="eiq-model-name">{label}</div>
                <div class="eiq-model-cat">{category}</div>
            </div>
            <div style="text-align:right;">
                <span class="eiq-badge {state}">{verdict}</span>
                <div class="eiq-mono" style="font-size:12px; color:var(--eiq-muted); margin-top:5px;">
                    {pct:.2f}% fraud prob.
                </div>
            </div>
        </div>
        <div class="eiq-mini-track">
            <div class="eiq-mini-fill" style="width:{pct:.1f}%; background:{fill_color};"></div>
        </div>
    </div>
    """)
 
 
def render_consensus(fraud_count, total):
    clear_count = total - fraud_count
    pct = (fraud_count / total * 100) if total else 0
    if total == 0:
        message = "No models were available to form a consensus."
    elif fraud_count == 0:
        message = "All available models agree this transaction looks clear."
    elif fraud_count == total:
        message = "All available models agree this transaction looks fraudulent."
    elif fraud_count > clear_count:
        message = "Most models lean toward flagging this transaction as fraud."
    elif fraud_count < clear_count:
        message = "Most models lean toward treating this transaction as clear."
    else:
        message = "Models are evenly split - manual review recommended."
 
    render_html(f"""
    <div class="eiq-glass-flat">
        <div class="eiq-small-label">Model Consensus</div>
        <div class="eiq-consensus-number">{fraud_count} / {total} models flagged this transaction as fraud</div>
        <div class="eiq-consensus-msg">{message}</div>
        <div class="eiq-prob-track">
            <div class="eiq-prob-fill fraud" style="width:{pct:.1f}%;"></div>
        </div>
    </div>
    """)
 
 
# ============================================================
# MAIN APP
# ============================================================
 
def main():
    init_session_state()
 
    models, model_errors = load_models()
    scaler, scaler_error = load_scaler()
    dataset, dataset_error = load_demo_dataset()
 
    models_ok_count = sum(1 for k in MODEL_REGISTRY if models.get(k) is not None)
    models_total = len(MODEL_REGISTRY)
    dataset_ok = dataset is not None
 
    render_header(models_ok_count, models_total, dataset_ok)
 
    with st.expander("System status", expanded=False):
        for key, meta in MODEL_REGISTRY.items():
            ok = models.get(key) is not None
            icon = "✅" if ok else "⚠️"
            note = "" if ok else f" — {model_errors.get(key, 'unavailable')}"
            st.write(f"{icon} {meta['label']}{note}")
        if scaler is None:
            st.write(f"⚠️ Scaler — {scaler_error}")
        else:
            st.write("✅ Scaler")
        if dataset is None:
            st.write(f"⚠️ Demo dataset — {dataset_error}")
        else:
            st.write(f"✅ Demo dataset — {len(dataset):,} transactions available")
 
    if scaler is None:
        st.error(
            "The feature scaler (models/scaler.pkl) could not be loaded, so no "
            "predictions can be made. Please check that the file exists."
        )
        return
 
    primary_model = models.get(PRIMARY_MODEL_KEY)
    if primary_model is None:
        st.error(
            "The primary model (Extra Trees) could not be loaded, so risk checks "
            "are unavailable right now. Other models may still work under "
            "'View Full Model Analysis' once a transaction is selected."
        )
 
    mode = st.radio(
        "Input mode",
        ["Demo Transaction", "Upload Transaction CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode != st.session_state.input_mode:
        st.session_state.input_mode = mode
        reset_transaction_state()
 
    render_html("<div style='height:8px'></div>")
 
    if mode == "Demo Transaction":
        render_demo_mode(dataset, dataset_error, models, model_errors, scaler)
    else:
        render_csv_mode(models, model_errors, scaler)
 
 
# ------------------------------------------------------------
# DEMO MODE
# ------------------------------------------------------------
 
def render_demo_mode(dataset, dataset_error, models, model_errors, scaler):
    if dataset is None:
        st.error(f"Demo mode is unavailable. {dataset_error}")
        return
 
    max_index = len(dataset) - 1
 
    # Determine current pipeline stage for the flow indicator.
    if st.session_state.show_full_analysis:
        stage = "full"
    elif st.session_state.transaction_checked:
        stage = "result"
    else:
        stage = "card"
    render_flow(stage)
 
    txn_number = st.number_input(
        "Transaction number",
        min_value=0,
        max_value=max_index,
        value=int(st.session_state.selected_transaction),
        step=1,
        help=f"Choose any transaction from 0 to {max_index}.",
    )
 
    if txn_number != st.session_state.selected_transaction:
        st.session_state.selected_transaction = txn_number
        reset_transaction_state()
 
    row = dataset.iloc[int(txn_number)]
    amount = float(row["Amount"])
 
    render_transaction_card(int(txn_number), amount, source="Demo dataset")
 
    c1, c2, c3 = st.columns(3)
    with c1:
        render_html(
            f"<div class='eiq-small-label'>Reference</div>"
            f"<div class='eiq-mono' style='font-weight:600;'>#{int(txn_number):06d}</div>"
        )
    with c2:
        render_html(
            f"<div class='eiq-small-label'>Amount</div>"
            f"<div class='eiq-mono' style='font-weight:600;'>${amount:,.2f}</div>"
        )
    with c3:
        render_html(
            "<div class='eiq-small-label'>Status</div>"
            "<div style='font-weight:600;'>Ready to check</div>"
        )
 
    render_html("<div style='height:14px'></div>")
 
    check_col, reset_col = st.columns([3, 1])
    with check_col:
        check_clicked = st.button("🔍 Check Transaction Risk", type="primary", use_container_width=True)
    with reset_col:
        if st.button("Reset", use_container_width=True):
            reset_transaction_state()
            st.rerun()
 
    primary_model = models.get(PRIMARY_MODEL_KEY)
 
    if check_clicked:
        if primary_model is None:
            st.error("Primary model (Extra Trees) is not available, so this check cannot run.")
        else:
            scan_placeholder = st.empty()
            with scan_placeholder.container():
                render_transaction_card(int(txn_number), amount, source="Demo dataset", scanning=True)
            with st.spinner("Scanning transaction and running AI model..."):
                try:
                    X = build_feature_frame(row, scaler)
                    pred, prob = fraud_probability(primary_model, X)
                    st.session_state.primary_result = {
                        "pred": pred,
                        "prob": prob,
                        "model_label": MODEL_REGISTRY[PRIMARY_MODEL_KEY]["label"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state.transaction_checked = True
                    st.session_state.show_full_analysis = False
                    st.session_state.all_model_results = None
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Prediction failed: {exc}")
                    st.session_state.transaction_checked = False
            scan_placeholder.empty()
            st.rerun()
 
    if st.session_state.transaction_checked and st.session_state.primary_result:
        result = st.session_state.primary_result
        render_primary_result(result["pred"], result["prob"], result["model_label"])
        st.caption(f"Checked at {result['timestamp']}")
 
        if st.button("📊 View Full Model Analysis", use_container_width=True):
            with st.spinner("Running all saved models..."):
                results, errors = run_all_models(models, row, scaler)
                st.session_state.all_model_results = results
                st.session_state.show_full_analysis = True
                if errors:
                    st.session_state["_full_analysis_errors"] = errors
            st.rerun()
 
    if st.session_state.show_full_analysis and st.session_state.all_model_results:
        render_full_analysis(st.session_state.all_model_results)
        errs = st.session_state.get("_full_analysis_errors")
        if errs:
            with st.expander("Some models could not be evaluated"):
                for key, msg in errs.items():
                    st.write(f"⚠️ {MODEL_REGISTRY[key]['label']}: {msg}")
 
 
def run_all_models(models, row, scaler):
    """Run every available model against a single transaction row.
    Returns (results_list, errors_dict). results_list items:
    {key, label, category, pred, prob}."""
    results = []
    errors = {}
    X = build_feature_frame(row, scaler)
    for key, meta in MODEL_REGISTRY.items():
        model = models.get(key)
        if model is None:
            errors[key] = model_errors_lookup(key)
            continue
        try:
            pred, prob = fraud_probability(model, X)
            results.append({
                "key": key,
                "label": meta["label"],
                "category": meta["category"],
                "pred": pred,
                "prob": prob,
            })
        except Exception as exc:  # noqa: BLE001
            errors[key] = str(exc)
    return results, errors
 
 
def model_errors_lookup(key):
    return "Model file could not be loaded."
 
 
def render_full_analysis(results):
    if not results:
        st.warning("No models were available to run a full analysis.")
        return
 
    total = len(results)
    fraud_count = sum(1 for r in results if r["pred"] == 1)
 
    render_html(
        "<div class='eiq-section-title'>🧩 Full Model Analysis "
        "<span class='eiq-tag'>ALL SAVED MODELS</span></div>"
    )
 
    for category, icon in [("Bagging", "🌳"), ("Boosting", "⚡")]:
        group = [r for r in results if r["category"] == category]
        if not group:
            continue
        render_html(
            f"<div class='eiq-small-label' style='margin-top:6px;'>{icon} {category} models</div>"
        )
        for r in group:
            render_model_row(r["label"], r["category"], r["pred"], r["prob"])
 
    render_consensus(fraud_count, total)
 
 
# ------------------------------------------------------------
# CSV UPLOAD MODE
# ------------------------------------------------------------
 
def render_csv_mode(models, model_errors, scaler):
    render_html("""
    <div class="eiq-glass-flat">
        <div class="eiq-small-label">Upload Transaction CSV</div>
        <div style="font-size:13px; color:var(--eiq-muted);">
            Upload a CSV containing the columns:
            <span class="eiq-mono" style="color:var(--eiq-text);">Time, V1...V28, Amount</span>
            (an optional <span class="eiq-mono" style="color:var(--eiq-text);">Class</span>
            column is ignored). Each row will be scored using the primary model.
        </div>
    </div>
    """)
 
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
 
    if uploaded is None:
        st.session_state.csv_results_df = None
        return
 
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the uploaded file: {exc}")
        return
 
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(
            "This CSV is missing required columns and cannot be scored:\n\n"
            + ", ".join(missing)
        )
        return
 
    primary_model = models.get(PRIMARY_MODEL_KEY)
    if primary_model is None:
        st.error("Primary model (Extra Trees) is not available, so CSV scoring cannot run.")
        return
 
    st.success(f"File accepted — {len(df):,} transactions found.")
 
    if st.button("🔍 Score All Transactions", type="primary", use_container_width=True):
        with st.spinner("Scanning transactions..."):
            try:
                X = build_feature_frame_batch(df, scaler)
                preds, probs = fraud_probability_batch(primary_model, X)
                results_df = df.copy()
                results_df["prediction"] = np.where(preds == 1, "FRAUD", "CLEAR")
                results_df["fraud_probability"] = probs
                results_df["risk_level"] = [risk_level(p) for p in probs]
                st.session_state.csv_results_df = results_df
                st.session_state.csv_selected_row = 0
                st.session_state.csv_show_row_analysis = False
            except Exception as exc:  # noqa: BLE001
                st.error(f"Batch prediction failed: {exc}")
                st.session_state.csv_results_df = None
 
    results_df = st.session_state.csv_results_df
    if results_df is None:
        return
 
    fraud_total = int((results_df["prediction"] == "FRAUD").sum())
    clear_total = int((results_df["prediction"] == "CLEAR").sum())
    total = len(results_df)
    fraud_pct = (fraud_total / total * 100) if total else 0
 
    render_html("<div class='eiq-section-title'>📈 Batch Summary</div>")
    m1, m2, m3 = st.columns(3)
    m1.metric("Transactions", f"{total:,}")
    m2.metric("Flagged as fraud", f"{fraud_total:,}")
    m3.metric("Fraud rate", f"{fraud_pct:.2f}%")
    st.caption(f"{clear_total:,} transactions predicted clear.")
 
    render_html("<div class='eiq-section-title' style='margin-top:16px;'>Results Preview</div>")
    preview_cols = ["prediction", "fraud_probability", "risk_level", "Amount"]
    st.dataframe(results_df[preview_cols].head(50), use_container_width=True)
 
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Prediction Results",
        data=csv_bytes,
        file_name=f"ensembleiq_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
 
    render_html("<div class='eiq-section-title' style='margin-top:16px;'>Inspect a Single Transaction</div>")
    row_index = st.number_input(
        "Row number",
        min_value=0,
        max_value=total - 1,
        value=int(st.session_state.csv_selected_row),
        step=1,
    )
    if row_index != st.session_state.csv_selected_row:
        st.session_state.csv_selected_row = row_index
        st.session_state.csv_show_row_analysis = False
 
    if st.button("📊 Full Model Analysis for This Row", use_container_width=True):
        row = results_df.iloc[int(row_index)]
        with st.spinner("Running all saved models..."):
            results, errors = run_all_models(models, row, scaler)
            st.session_state.csv_row_results = results
            st.session_state.csv_row_errors = errors
            st.session_state.csv_show_row_analysis = True
 
    if st.session_state.csv_show_row_analysis:
        render_full_analysis(st.session_state.get("csv_row_results", []))
        errs = st.session_state.get("csv_row_errors")
        if errs:
            with st.expander("Some models could not be evaluated"):
                for key, msg in errs.items():
                    st.write(f"⚠️ {MODEL_REGISTRY[key]['label']}: {msg}")
 
 
if __name__ == "__main__":
    main()
