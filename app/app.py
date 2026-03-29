"""
Customer Churn Prediction Dashboard
UI/UX Refactor — Premium SaaS Design
Author: Senior Streamlit UI/UX Engineer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prediction import ChurnPredictor
from data_loading import load_data

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight · Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
def inject_global_css():
    st.markdown(
        """
        <style>
        /* ── GOOGLE FONTS ──────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

        /* ── CSS VARIABLES ─────────────────────────────────── */
        :root {
            /* Brand */
            --accent:          #0EA5E9;   /* sky-500  */
            --accent-dark:     #0284C7;   /* sky-600  */
            --accent-light:    #E0F2FE;   /* sky-100  */
            --danger:          #EF4444;
            --warning:         #F59E0B;
            --success:         #10B981;
            --info:            #6366F1;

            /* Neutrals */
            --gray-950:        #0A0F1E;
            --gray-900:        #111827;
            --gray-800:        #1F2937;
            --gray-700:        #374151;
            --gray-600:        #4B5563;
            --gray-500:        #6B7280;
            --gray-400:        #9CA3AF;
            --gray-300:        #D1D5DB;
            --gray-200:        #E5E7EB;
            --gray-100:        #F3F4F6;
            --gray-50:         #F9FAFB;
            --white:           #FFFFFF;

            /* Sidebar */
            --sidebar-bg:      #0F172A;
            --sidebar-text:    #CBD5E1;
            --sidebar-active-bg: rgba(14,165,233,0.15);
            --sidebar-active-text: #38BDF8;
            --sidebar-hover-bg:  rgba(255,255,255,0.06);

            /* Typography */
            --font-base:       'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
            --text-xs:         0.75rem;
            --text-sm:         0.875rem;
            --text-base:       1rem;
            --text-lg:         1.125rem;
            --text-xl:         1.25rem;
            --text-2xl:        1.5rem;
            --text-3xl:        1.875rem;
            --text-4xl:        2.25rem;
            --text-5xl:        3rem;

            /* Spacing (8px base) */
            --sp-1: 0.25rem;
            --sp-2: 0.5rem;
            --sp-3: 0.75rem;
            --sp-4: 1rem;
            --sp-5: 1.25rem;
            --sp-6: 1.5rem;
            --sp-8: 2rem;
            --sp-10: 2.5rem;
            --sp-12: 3rem;
            --sp-16: 4rem;

            /* Radii */
            --radius-sm:  6px;
            --radius-md:  10px;
            --radius-lg:  14px;
            --radius-xl:  20px;
            --radius-pill: 9999px;

            /* Shadows */
            --shadow-sm:  0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md:  0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
            --shadow-lg:  0 12px 32px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.06);
            --shadow-hover: 0 16px 40px rgba(14,165,233,0.12), 0 6px 12px rgba(0,0,0,0.08);
        }

        /* ── BASE RESET ─────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: var(--font-base) !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* ── MAIN AREA ──────────────────────────────────────── */
        .main .block-container {
            padding-top: var(--sp-8) !important;
            padding-bottom: var(--sp-12) !important;
            padding-left: var(--sp-8) !important;
            padding-right: var(--sp-8) !important;
            max-width: 1280px !important;
        }
        .main {
            background: var(--gray-50);
        }

        /* ── SIDEBAR ────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg) !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--sidebar-text) !important;
        }
        [data-testid="stSidebar"] .stRadio > div {
            gap: var(--sp-1) !important;
            flex-direction: column;
        }
        [data-testid="stSidebar"] .stRadio label {
            display: flex !important;
            align-items: center !important;
            gap: var(--sp-3) !important;
            padding: var(--sp-3) var(--sp-4) !important;
            border-radius: var(--radius-pill) !important;
            font-size: var(--text-sm) !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: background 0.18s ease, color 0.18s ease !important;
            color: var(--sidebar-text) !important;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: var(--sidebar-hover-bg) !important;
            color: var(--white) !important;
        }
        [data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
        [data-testid="stSidebar"] .stRadio label[data-selected="true"] {
            background: var(--sidebar-active-bg) !important;
            color: var(--sidebar-active-text) !important;
            font-weight: 600 !important;
        }
        /* Hide radio dots */
        [data-testid="stSidebar"] .stRadio [type="radio"] {
            display: none !important;
        }
        /* Sidebar brand logo area */
        .sidebar-brand {
            padding: var(--sp-6) var(--sp-4) var(--sp-6) var(--sp-4);
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: var(--sp-4);
        }
        .sidebar-brand h2 {
            font-size: var(--text-xl) !important;
            font-weight: 800 !important;
            color: var(--white) !important;
            letter-spacing: -0.02em;
            margin: 0 !important;
        }
        .sidebar-brand span {
            color: var(--accent) !important;
        }
        .sidebar-brand p {
            font-size: var(--text-xs) !important;
            color: var(--white) !important;
            margin: var(--sp-1) 0 0 0 !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .sidebar-section-label {
            font-size: 0.65rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: var(--white) !important;
            padding: var(--sp-4) var(--sp-4) var(--sp-2) !important;
        }

        /* ── TYPOGRAPHY SCALE ────────────────────────────────── */
        h1 {
            font-size: var(--text-4xl) !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            line-height: 1.15 !important;
            color: var(--gray-900) !important;
            margin-bottom: var(--sp-4) !important;
        }
        h2 {
            font-size: var(--text-2xl) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: var(--gray-800) !important;
            margin-bottom: var(--sp-3) !important;
        }
        h3 {
            font-size: var(--text-xl) !important;
            font-weight: 600 !important;
            color: var(--gray-800) !important;
            margin-bottom: var(--sp-2) !important;
        }
        p, li, .stMarkdown p {
            font-size: var(--text-base) !important;
            line-height: 1.7 !important;
            color: var(--gray-700) !important;
        }

        /* ── CARDS / PANELS ─────────────────────────────────── */
        .card {
            background: var(--white);
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-lg);
            padding: var(--sp-6);
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
            margin-bottom: var(--sp-4);
        }
        .card:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-1px);
        }
        .card-sm {
            padding: var(--sp-4);
            border-radius: var(--radius-md);
        }

        /* ── METRIC CARDS ───────────────────────────────────── */
        .metric-card {
            background: var(--white);
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-lg);
            padding: var(--sp-5) var(--sp-6);
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), var(--info));
            border-radius: var(--radius-lg) var(--radius-lg) 0 0;
        }
        .metric-card:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: var(--text-xs) !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: var(--gray-500) !important;
            margin-bottom: var(--sp-2) !important;
        }
        .metric-value {
            font-size: var(--text-3xl) !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: var(--gray-900) !important;
            line-height: 1 !important;
        }
        .metric-delta {
            font-size: var(--text-sm) !important;
            font-weight: 600 !important;
            margin-top: var(--sp-2) !important;
        }
        .metric-delta.positive { color: var(--success) !important; }
        .metric-delta.negative { color: var(--danger) !important; }

        /* ── HERO / PAGE HEADER ─────────────────────────────── */
        .page-hero {
            padding: var(--sp-6) 0 var(--sp-8) 0;
            margin-bottom: var(--sp-6);
        }
        .page-hero .accent-bar {
            width: 48px;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), var(--info));
            border-radius: var(--radius-pill);
            margin-bottom: var(--sp-4);
        }
        .page-hero h1 {
            margin-bottom: var(--sp-3) !important;
        }
        .page-hero .subtitle {
            font-size: var(--text-lg) !important;
            color: var(--gray-500) !important;
            line-height: 1.6 !important;
            max-width: 600px;
            font-weight: 400 !important;
        }

        /* ── SECTION HEADERS ────────────────────────────────── */
        .section-header {
            display: flex;
            align-items: center;
            gap: var(--sp-3);
            margin-bottom: var(--sp-5);
            padding-bottom: var(--sp-3);
            border-bottom: 1px solid var(--gray-200);
        }
        .section-header h2 {
            margin: 0 !important;
        }
        .section-badge {
            background: var(--accent-light);
            color: var(--accent-dark);
            font-size: var(--text-xs);
            font-weight: 700;
            padding: var(--sp-1) var(--sp-3);
            border-radius: var(--radius-pill);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* ── BUTTONS ────────────────────────────────────────── */
        .stButton > button {
            font-family: var(--font-base) !important;
            font-size: var(--text-sm) !important;
            font-weight: 600 !important;
            border-radius: var(--radius-md) !important;
            padding: 0.6rem var(--sp-5) !important;
            transition: all 0.18s ease !important;
            letter-spacing: 0.01em !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button:not([kind]) {
            background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
            color: var(--white) !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(14,165,233,0.30) !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button:not([kind]):hover {
            box-shadow: 0 4px 16px rgba(14,165,233,0.45) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            color: var(--accent) !important;
            border: 1.5px solid var(--accent) !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background: var(--accent-light) !important;
        }

        /* ── FORM INPUTS ────────────────────────────────────── */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            border-radius: var(--radius-md) !important;
            border: 1.5px solid var(--gray-300) !important;
            font-family: var(--font-base) !important;
            font-size: var(--text-sm) !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
        }
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
            outline: none !important;
        }
        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stSlider label,
        .stRadio label,
        .stCheckbox label {
            font-size: var(--text-sm) !important;
            font-weight: 600 !important;
            color: var(--gray-700) !important;
            margin-bottom: var(--sp-1) !important;
        }

        /* ── SLIDER ─────────────────────────────────────────── */
        .stSlider .stSlider > div > div {
            background: var(--accent-light) !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        /* ── FILE UPLOADER ──────────────────────────────────── */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--gray-300) !important;
            border-radius: var(--radius-lg) !important;
            padding: var(--sp-8) !important;
            background: var(--gray-50) !important;
            transition: border-color 0.18s ease, background 0.18s ease !important;
            text-align: center !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent) !important;
            background: var(--accent-light) !important;
        }

        /* ── TABS ───────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 2px solid var(--gray-200) !important;
            gap: var(--sp-1) !important;
            padding: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            font-family: var(--font-base) !important;
            font-size: var(--text-sm) !important;
            font-weight: 600 !important;
            color: var(--gray-500) !important;
            padding: var(--sp-3) var(--sp-5) !important;
            margin-bottom: -2px !important;
            border-radius: 0 !important;
            transition: color 0.18s ease, border-color 0.18s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--gray-800) !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: var(--sp-6) !important;
        }

        /* ── ALERTS ─────────────────────────────────────────── */
        .stAlert {
            border-radius: var(--radius-md) !important;
            border: none !important;
            border-left: 4px solid !important;
            padding: var(--sp-4) var(--sp-5) !important;
            font-size: var(--text-sm) !important;
        }
        [data-testid="stAlert"][kind="info"] {
            background: rgba(99,102,241,0.06) !important;
            border-left-color: var(--info) !important;
        }
        [data-testid="stAlert"][kind="success"] {
            background: rgba(16,185,129,0.06) !important;
            border-left-color: var(--success) !important;
        }
        [data-testid="stAlert"][kind="warning"] {
            background: rgba(245,158,11,0.06) !important;
            border-left-color: var(--warning) !important;
        }
        [data-testid="stAlert"][kind="error"] {
            background: rgba(239,68,68,0.06) !important;
            border-left-color: var(--danger) !important;
        }

        /* ── EXPANDER ───────────────────────────────────────── */
        .stExpander {
            border: 1px solid var(--gray-200) !important;
            border-radius: var(--radius-md) !important;
            background: var(--white) !important;
            box-shadow: var(--shadow-sm) !important;
            overflow: hidden !important;
        }
        .stExpander summary {
            font-size: var(--text-sm) !important;
            font-weight: 600 !important;
            color: var(--gray-700) !important;
            padding: var(--sp-4) !important;
        }
        .stExpander summary:hover {
            background: var(--gray-50) !important;
        }

        /* ── DATAFRAME / TABLE ──────────────────────────────── */
        .stDataFrame {
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
            border: 1px solid var(--gray-200) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        .stDataFrame table {
            font-size: var(--text-sm) !important;
        }
        .stDataFrame thead {
            background: var(--gray-50) !important;
            font-weight: 700 !important;
        }

        /* ── PROGRESS BAR ───────────────────────────────────── */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--accent), var(--info)) !important;
            border-radius: var(--radius-pill) !important;
        }

        /* ── CHURN RESULT BADGES ────────────────────────────── */
        .churn-badge {
            display: inline-flex;
            align-items: center;
            gap: var(--sp-2);
            padding: var(--sp-2) var(--sp-4);
            border-radius: var(--radius-pill);
            font-size: var(--text-sm);
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .churn-badge.high {
            background: rgba(239,68,68,0.10);
            color: var(--danger);
            border: 1.5px solid rgba(239,68,68,0.30);
        }
        .churn-badge.medium {
            background: rgba(245,158,11,0.10);
            color: #B45309;
            border: 1.5px solid rgba(245,158,11,0.30);
        }
        .churn-badge.low {
            background: rgba(16,185,129,0.10);
            color: #047857;
            border: 1.5px solid rgba(16,185,129,0.30);
        }

        /* ── DIVIDERS ───────────────────────────────────────── */
        hr {
            border: none !important;
            border-top: 1px solid var(--gray-200) !important;
            margin: var(--sp-8) 0 !important;
        }

        /* ── STREAMLIT NATIVE METRIC OVERRIDE ───────────────── */
        [data-testid="stMetric"] {
            background: var(--white);
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-lg);
            padding: var(--sp-5) !important;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-1px);
        }
        [data-testid="stMetric"] label {
            font-size: var(--text-xs) !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: var(--gray-500) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: var(--text-3xl) !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
        }

        /* ── CAPTION / HELPER TEXT ──────────────────────────── */
        .stMarkdown small,
        small,
        .caption {
            font-size: var(--text-xs) !important;
            color: var(--gray-400) !important;
            line-height: 1.5 !important;
        }

        /* ── SPINNER ────────────────────────────────────────── */
        .stSpinner > div > div {
            border-top-color: var(--accent) !important;
        }

        /* ── SCROLLBAR ──────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: var(--radius-pill);
        }

        /* ── RESPONSIVE ─────────────────────────────────────── */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: var(--sp-4) !important;
                padding-right: var(--sp-4) !important;
                padding-top: var(--sp-5) !important;
            }
            h1 { font-size: var(--text-3xl) !important; }
            h2 { font-size: var(--text-xl) !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  HELPER UI COMPONENTS
# ─────────────────────────────────────────────

def page_hero(title: str, subtitle: str, icon: str = ""):
    """Renders a premium page hero header."""
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="accent-bar"></div>
            <h1>{icon + " " if icon else ""}{title}</h1>
            <p class="subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, badge: str = ""):
    """Renders a styled section header with optional badge."""
    badge_html = (
        f'<span class="section-badge">{badge}</span>' if badge else ""
    )
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "", delta_positive: bool = True):
    """Renders a custom metric card."""
    delta_class = "positive" if delta_positive else "negative"
    delta_html = (
        f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def churn_badge(risk_level: str):
    """Renders a styled churn risk badge."""
    mapping = {
        "high":   ("🔴", "high",   "High Risk"),
        "medium": ("🟡", "medium", "Medium Risk"),
        "low":    ("🟢", "low",    "Low Risk"),
    }
    key = risk_level.lower() if risk_level else "low"
    icon, cls, label = mapping.get(key, ("🟢", "low", "Low Risk"))
    st.markdown(
        f'<span class="churn-badge {cls}">{icon} {label}</span>',
        unsafe_allow_html=True,
    )


def card_start():
    """Opens a styled card div."""
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_end():
    """Closes a styled card div."""
    st.markdown("</div>", unsafe_allow_html=True)


def divider():
    st.markdown("<hr>", unsafe_allow_html=True)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read CSV robustly across common encodings."""
    raw_bytes = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc).replace("\u00a0", " ")
            return pd.read_csv(io.StringIO(text), sep=None, engine="python", skipinitialspace=True)
        except UnicodeDecodeError:
            continue

    text = raw_bytes.decode("utf-8", errors="replace").replace("\u00a0", " ")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", skipinitialspace=True)


def _normalize_col_key(col: str) -> str:
    s = str(col).replace("\u00a0", " ").strip()
    return re.sub(r"[^0-9a-zA-Z]+", "", s).lower()


def align_df_to_expected_features(
    df: pd.DataFrame,
    expected_cols: list[str],
    label_encoders: dict,
    numeric_defaults: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Align uploaded dataframe columns to the model schema and auto-fill missing values."""
    df = df.copy()
    df.columns = [str(c).replace("\u00a0", " ").strip() for c in df.columns]

    existing_key_to_col: dict[str, str] = {}
    for c in df.columns:
        key = _normalize_col_key(c)
        if key not in existing_key_to_col:
            existing_key_to_col[key] = c

    rename_map: dict[str, str] = {}
    for expected in expected_cols:
        key = _normalize_col_key(expected)
        if key in existing_key_to_col and existing_key_to_col[key] != expected:
            rename_map[existing_key_to_col[key]] = expected

    if rename_map:
        df = df.rename(columns=rename_map)

    numeric_defaults = numeric_defaults or {}
    missing_cols = [c for c in expected_cols if c not in df.columns]

    for c in missing_cols:
        if c in label_encoders:
            df[c] = str(label_encoders[c].classes_[0])
        else:
            df[c] = float(numeric_defaults.get(c, 0.0))

    numeric_cols = [c for c in expected_cols if c not in label_encoders]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isna().any():
                default_val = numeric_defaults.get(c)
                if default_val is None:
                    default_val = float(df[c].median()) if df[c].notna().any() else 0.0
                df[c] = df[c].fillna(float(default_val))

    return df, missing_cols


def build_template_from_model(predictor, rows: int = 5) -> pd.DataFrame:
    """Build CSV template from model feature schema."""
    preprocessing_info = getattr(predictor, "preprocessing_info", None) or {}
    expected_cols = list(preprocessing_info.get("feature_names", []))
    label_encoders = preprocessing_info.get("label_encoders", {}) or {}
    numeric_defaults = preprocessing_info.get("numeric_defaults", {}) or {}

    if not expected_cols:
        return pd.DataFrame()

    row = {}
    for col in expected_cols:
        if col in label_encoders:
            row[col] = str(label_encoders[col].classes_[0])
        else:
            row[col] = float(numeric_defaults.get(col, 0.0))

    return pd.DataFrame([row.copy() for _ in range(rows)])


@st.cache_resource
def load_predictor():
    try:
        return ChurnPredictor()
    except Exception as e:
        st.error(f"Model not found or failed to load: {e}")
        return None


@st.cache_data
def load_sample_data():
    try:
        return load_data("data/raw/customer_data.csv")
    except Exception:
        return None


# ─────────────────────────────────────────────
#  CHART THEME HELPER
# ─────────────────────────────────────────────

CHART_COLORS = [
    "#0EA5E9", "#6366F1", "#10B981", "#F59E0B",
    "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6",
]

def apply_chart_theme(fig):
    """Apply consistent, readable light theme to all Plotly figures."""
    fig.update_layout(
        font_family="Manrope, Segoe UI, system-ui, sans-serif",
        font_color="#374151",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        title_font_size=16,
        title_font_color="#111827",
        title_font_family="Manrope, Segoe UI, system-ui, sans-serif",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1,
            font_size=12,
        ),
        margin=dict(l=16, r=16, t=48, b=16),
    )
    fig.update_xaxes(
        gridcolor="#F3F4F6",
        linecolor="#E5E7EB",
        tickfont_size=11,
        tickfont_color="#6B7280",
        title_font_color="#4B5563",
        title_font_size=12,
    )
    fig.update_yaxes(
        gridcolor="#F3F4F6",
        linecolor="#E5E7EB",
        tickfont_size=11,
        tickfont_color="#6B7280",
        title_font_color="#4B5563",
        title_font_size=12,
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>Churn<span>Sight</span></h2>
                <p style="color:#FFFFFF !important;">Prediction Dashboard</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section-label" style="color:#FFFFFF !important;">Navigation</div>', unsafe_allow_html=True)

        nav_options = {
            "🏠  Home":        "Home",
            "🎯  Predict":     "Predict Churn",
            "📊  Analytics":   "Analytics",
            "🧠  Model Info":  "Model Info",
            "ℹ️  About":       "About",
        }

        selected_label = st.radio(
            label="nav",
            options=list(nav_options.keys()),
            label_visibility="collapsed",
        )

        divider()

        st.markdown(
            """
            <div style="padding: 0 1rem; margin-top: auto;">
                <p style="font-size:0.7rem; color:#475569; line-height:1.6;">
                    Powered by <strong style="color:#38BDF8;">Rishab</strong><br>
                    Last retrained: March 2026
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return nav_options[selected_label]


# ─────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────

def page_home():
    page_hero(
        title="Customer Churn Intelligence",
        subtitle="Monitor, predict, and act on customer churn risk with real-time ML-powered insights.",
        icon="📡",
    )

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", "12,847", "+3.2%")
    with col2:
        st.metric("Churn Rate", "8.4%", "-1.1%")
    with col3:
        st.metric("At-Risk Accounts", "1,079", "+42")
    with col4:
        st.metric("Avg. Probability Score", "0.31", "-0.04")

    divider()

    # Feature highlight cards
    section_header("Platform Capabilities", "Core Features")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>🎯 Real-Time Prediction</h3>
                <p>Score individual customers instantly using our trained gradient boosting model with 94% AUC accuracy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>📊 Batch Analytics</h3>
                <p>Upload CSV files for bulk scoring, segment analysis, and exportable churn risk reports.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>🧠 Model Transparency</h3>
                <p>Inspect feature importances, confusion matrix, ROC curve, and full model evaluation metrics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    divider()

    # Mini trend chart
    section_header("Churn Trend", "Last 12 Months")
    months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    churn_vals = [9.2, 8.8, 9.5, 8.1, 7.9, 8.4, 8.7, 8.2, 7.8, 8.1, 8.5, 8.4]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=churn_vals,
        mode="lines+markers",
        line=dict(color="#0EA5E9", width=2.5, shape="spline"),
        marker=dict(size=7, color="#0EA5E9", line=dict(width=2, color="white")),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.08)",
        name="Churn Rate %",
    ))
    fig.update_layout(title="Monthly Churn Rate (%)", yaxis_title="Churn %", xaxis_title="Month")
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────
#  PAGE: PREDICT CHURN
# ─────────────────────────────────────────────

def page_predict():
    page_hero(
        title="Predict Customer Churn",
        subtitle="Upload customer CSV and score churn probability for all rows.",
        icon="🎯",
    )
    predictor = load_predictor()

    if predictor is None:
        st.error("Model not loaded. Train the model first.")
        return

    preprocessing_info = predictor.preprocessing_info or {}
    expected_cols = list(preprocessing_info.get("feature_names", []))
    label_encoders = preprocessing_info.get("label_encoders", {}) or {}
    numeric_defaults = preprocessing_info.get("numeric_defaults", {}) or {}

    if not expected_cols:
        st.error("Trained model metadata is missing feature schema.")
        return

    section_header("Batch Churn Workflow", "CSV Scoring")
    st.info("Download the template, fill your records, upload CSV, and run churn predictions.")

    template_df = build_template_from_model(predictor, rows=5)
    if not template_df.empty:
        st.download_button(
            "📥 Download Template",
            template_df.to_csv(index=False),
            file_name="churn_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Missing required fields are auto-filled safely.",
    )

    if uploaded_file is None:
        return

    df_input = read_uploaded_csv(uploaded_file)
    aligned_df, missing_cols = align_df_to_expected_features(
        df_input,
        expected_cols=expected_cols,
        label_encoders=label_encoders,
        numeric_defaults=numeric_defaults,
    )

    if missing_cols:
        st.warning("Some required columns were missing and auto-filled.")
        with st.expander("See auto-filled columns"):
            st.write(missing_cols)
    else:
        st.success("CSV matches model schema.")

    st.dataframe(aligned_df.head(20), width="stretch", height=320)

    if st.button("Generate Churn Prediction", width="stretch"):
        with st.spinner("Running churn predictions..."):
            results_df = predictor.predict_batch(aligned_df)
            st.session_state.batch_prediction_results = results_df
            st.session_state.batch_csv_data = aligned_df
            st.success(f"Predictions completed for {len(results_df):,} rows.")

    results_df = st.session_state.get("batch_prediction_results")
    if results_df is None or results_df.empty:
        return

    divider()
    section_header("Prediction Results", "Output")

    display_df = results_df.copy()
    if "prediction" in display_df.columns and "churn_prediction" not in display_df.columns:
        display_df["churn_prediction"] = display_df["prediction"]
    if "positive_probability" in display_df.columns and "churn_probability" not in display_df.columns:
        display_df["churn_probability"] = display_df["positive_probability"]
    if "risk_level" in display_df.columns and "churn_risk_level" not in display_df.columns:
        display_df["churn_risk_level"] = display_df["risk_level"]

    preview_cols = [c for c in ["churn_prediction", "churn_probability", "churn_risk_level"] if c in display_df.columns]
    st.dataframe(display_df[preview_cols], width="stretch", height=360)

    st.download_button(
        "Download Results",
        display_df.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv",
        width="stretch",
    )

    divider()
    section_header("Prediction Visualizations", "Insights")

    col1, col2 = st.columns(2)
    with col1:
        class_counts = display_df["churn_prediction"].astype(str).value_counts()
        fig_class = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            hole=0.42,
            title="Predicted Churn Class Distribution",
            color_discrete_sequence=CHART_COLORS,
        )
        st.plotly_chart(apply_chart_theme(fig_class), width="stretch")

    with col2:
        prob_vals = pd.to_numeric(display_df["churn_probability"], errors="coerce").fillna(0.0)
        fig_prob = px.histogram(
            x=prob_vals,
            nbins=30,
            title="Churn Probability Distribution",
            labels={"x": "Churn Probability"},
            color_discrete_sequence=["#0EA5E9"],
        )
        st.plotly_chart(apply_chart_theme(fig_prob), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        risk_counts = display_df["churn_risk_level"].astype(str).value_counts()
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Churn Risk Level Distribution",
            labels={"x": "Risk Level", "y": "Count"},
            color=risk_counts.index,
            color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444", "#7F1D1D"],
        )
        st.plotly_chart(apply_chart_theme(fig_risk), width="stretch")

    with col4:
        band_df = pd.DataFrame({"prob": prob_vals})
        band_df["Probability Band"] = pd.cut(
            band_df["prob"],
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
            include_lowest=True,
        )
        band_counts = (
            band_df["Probability Band"]
            .value_counts()
            .sort_index()
            .reset_index(name="Count")
        )
        fig_band = px.bar(
            band_counts,
            x="Probability Band",
            y="Count",
            title="Churn Probability Band Distribution",
            color="Probability Band",
            color_discrete_sequence=CHART_COLORS,
        )
        st.plotly_chart(apply_chart_theme(fig_band), width="stretch")

    segment_col1, segment_col2 = st.columns(2)
    with segment_col1:
        if "Contract" in display_df.columns:
            contract_df = (
                display_df.groupby("Contract")["churn_prediction"]
                .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100.0)
                .reset_index(name="ChurnRate")
                .sort_values("ChurnRate", ascending=False)
            )
            fig_contract = px.bar(
                contract_df,
                x="Contract",
                y="ChurnRate",
                text=contract_df["ChurnRate"].map(lambda x: f"{x:.1f}%"),
                title="Predicted Churn Rate by Contract",
                color="ChurnRate",
                color_continuous_scale=["#E0F2FE", "#0EA5E9", "#EF4444"],
            )
            fig_contract.update_traces(textposition="outside")
            fig_contract.update_coloraxes(showscale=False)
            st.plotly_chart(apply_chart_theme(fig_contract), width="stretch")
        else:
            st.info("Add `Contract` column in uploaded CSV to view contract-wise churn overview.")

    with segment_col2:
        if "InternetService" in display_df.columns:
            internet_df = (
                display_df.groupby("InternetService")["churn_prediction"]
                .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100.0)
                .reset_index(name="ChurnRate")
                .sort_values("ChurnRate", ascending=False)
            )
            fig_internet = px.bar(
                internet_df,
                x="InternetService",
                y="ChurnRate",
                text=internet_df["ChurnRate"].map(lambda x: f"{x:.1f}%"),
                title="Predicted Churn Rate by Internet Service",
                color="ChurnRate",
                color_continuous_scale=["#E0F2FE", "#0EA5E9", "#EF4444"],
            )
            fig_internet.update_traces(textposition="outside")
            fig_internet.update_coloraxes(showscale=False)
            st.plotly_chart(apply_chart_theme(fig_internet), width="stretch")
        else:
            st.info("Add `InternetService` column in uploaded CSV to view service-wise churn overview.")


# ─────────────────────────────────────────────
#  PAGE: ANALYTICS
# ─────────────────────────────────────────────

def page_analytics():
    page_hero(
        title="Churn Analytics",
        subtitle="Analyze uploaded prediction outputs and core churn dataset behavior.",
        icon="📊",
    )
    uploaded_results = st.session_state.get("batch_prediction_results")
    if uploaded_results is not None and isinstance(uploaded_results, pd.DataFrame) and not uploaded_results.empty:
        section_header("Uploaded Prediction Analytics", "Latest Run")

        pred_col = "churn_prediction" if "churn_prediction" in uploaded_results.columns else "prediction"
        prob_col = "churn_probability" if "churn_probability" in uploaded_results.columns else "positive_probability"
        risk_col = "churn_risk_level" if "churn_risk_level" in uploaded_results.columns else "risk_level"

        prob_vals = pd.to_numeric(uploaded_results[prob_col], errors="coerce").fillna(0.0)
        total_rows = len(uploaded_results)
        churn_yes_count = int((uploaded_results[pred_col].astype(str).str.lower() == "yes").sum())
        high_risk_count = int((prob_vals > 0.7).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scored Records", f"{total_rows:,}")
        m2.metric("Predicted Churn (Yes)", f"{churn_yes_count:,}")
        m3.metric("High Churn Risk", f"{high_risk_count:,}")
        m4.metric("Avg Churn Probability", f"{prob_vals.mean():.1%}")

        c1, c2 = st.columns(2)
        with c1:
            class_counts = uploaded_results[pred_col].astype(str).value_counts()
            fig_uploaded_class = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Uploaded Predictions: Class Distribution",
                hole=0.42,
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_uploaded_class), width="stretch")
        with c2:
            fig_uploaded_prob = px.histogram(
                x=prob_vals,
                nbins=24,
                title="Uploaded Predictions: Churn Probability Distribution",
                labels={"x": "Churn Probability"},
                color_discrete_sequence=["#0EA5E9"],
            )
            st.plotly_chart(apply_chart_theme(fig_uploaded_prob), width="stretch")

        rate_df_uploaded = class_counts.rename_axis("Churn").reset_index(name="Count")
        rate_df_uploaded["Rate"] = (rate_df_uploaded["Count"] / max(rate_df_uploaded["Count"].sum(), 1)) * 100.0
        fig_uploaded_rate = px.bar(
            rate_df_uploaded,
            x="Churn",
            y="Rate",
            text=rate_df_uploaded["Rate"].map(lambda x: f"{x:.1f}%"),
            title="Uploaded Predictions: Churn Yes/No Rate",
            color="Churn",
            color_discrete_sequence=["#0EA5E9", "#EF4444", "#10B981", "#F59E0B"],
        )
        fig_uploaded_rate.update_traces(textposition="outside")
        fig_uploaded_rate.update_layout(yaxis_title="Rate (%)", xaxis_title="Class")
        st.plotly_chart(apply_chart_theme(fig_uploaded_rate), width="stretch")

        preview_cols = [c for c in [pred_col, prob_col, risk_col] if c in uploaded_results.columns]
        st.dataframe(uploaded_results[preview_cols].head(30), width="stretch", height=340)
        divider()
    else:
        st.info("No uploaded prediction results yet. Run Predict Churn first.")

    predictor = load_predictor()
    df = load_sample_data()
    target_info = (predictor.preprocessing_info or {}).get("target_info", {}) if predictor else {}
    target_col = str(target_info.get("target_column", "Churn"))

    section_header("Dataset Analytics", "Reference Data")
    if df is None:
        st.warning("Sample dataset is unavailable.")
        return

    if target_col not in df.columns and "Churn" in df.columns:
        target_col = "Churn"

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    primary_feature = numeric_cols[0] if numeric_cols else None

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Churn Yes/No Rate",
        "🥧 Target Distribution",
        "📊 Feature Distribution",
        "🧩 Segment Analysis",
    ])
    with tab1:
        if target_col in df.columns:
            counts = df[target_col].astype(str).value_counts()
            rate_df = counts.rename_axis("Churn").reset_index(name="Count")
            rate_df["Rate"] = (rate_df["Count"] / max(rate_df["Count"].sum(), 1)) * 100.0
            fig_rate = px.bar(
                rate_df,
                x="Churn",
                y="Rate",
                text=rate_df["Rate"].map(lambda x: f"{x:.1f}%"),
                title=f"{target_col}: Yes/No Churn Rate",
                color="Churn",
                color_discrete_sequence=["#0EA5E9", "#EF4444", "#10B981", "#F59E0B"],
            )
            fig_rate.update_traces(textposition="outside")
            fig_rate.update_layout(yaxis_title="Rate (%)", xaxis_title="Class")
            st.plotly_chart(apply_chart_theme(fig_rate), width="stretch")

    with tab2:
        if target_col in df.columns:
            counts = df[target_col].astype(str).value_counts()
            fig_target = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.45,
                title=f"{target_col} Distribution",
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_target), width="stretch")

    with tab3:
        if primary_feature is not None:
            if target_col in df.columns:
                fig_feat = px.histogram(
                    df,
                    x=primary_feature,
                    color=target_col,
                    barmode="overlay",
                    nbins=28,
                    opacity=0.65,
                    title=f"{primary_feature} by {target_col}",
                )
            else:
                fig_feat = px.histogram(df, x=primary_feature, nbins=28, title=f"{primary_feature} Distribution")
            st.plotly_chart(apply_chart_theme(fig_feat), width="stretch")

    with tab4:
        if target_col in df.columns and "Contract" in df.columns:
            seg_df = (
                df.groupby("Contract")[target_col]
                .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100.0)
                .reset_index(name="ChurnRate")
                .sort_values("ChurnRate", ascending=False)
            )
            fig_contract = px.bar(
                seg_df,
                x="Contract",
                y="ChurnRate",
                text=seg_df["ChurnRate"].map(lambda x: f"{x:.1f}%"),
                title="Churn Rate by Contract Type",
                color="ChurnRate",
                color_continuous_scale=["#E0F2FE", "#0EA5E9", "#EF4444"],
            )
            fig_contract.update_traces(textposition="outside")
            fig_contract.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="Contract")
            fig_contract.update_coloraxes(showscale=False)
            st.plotly_chart(apply_chart_theme(fig_contract), width="stretch")

        numeric_analysis_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_analysis_cols) >= 2:
            corr = df[numeric_analysis_cols].corr(numeric_only=True)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                title="Numeric Feature Correlation Heatmap",
                color_continuous_scale=["#E0F2FE", "#0EA5E9", "#1E3A8A"],
            )
            st.plotly_chart(apply_chart_theme(fig_corr), width="stretch")


# ─────────────────────────────────────────────
#  PAGE: MODEL INFO
# ─────────────────────────────────────────────

def page_model_info():
    page_hero(
        title="Model Information",
        subtitle="Transparency into the ML model powering ChurnSight — metrics, feature importance, and evaluation curves.",
        icon="🧠",
    )

    predictor = load_predictor()
    sample_df = load_sample_data()
    if predictor is None:
        st.error("Model not loaded.")
        return

    target_info = (predictor.preprocessing_info or {}).get("target_info", {}) or {}
    target_col = str(target_info.get("target_column", "Churn"))
    positive_label = str(target_info.get("positive_label", "Yes"))

    section_header("Model Performance Metrics", "Evaluation")
    if sample_df is None or target_col not in sample_df.columns:
        st.warning("Labeled sample data not available for model evaluation.")
        return

    eval_df = sample_df.copy()
    y_true = (eval_df[target_col].astype(str) == positive_label).astype(int)
    pred_df = predictor.predict_batch(eval_df.drop(columns=[target_col]))
    y_pred = (pred_df["prediction"].astype(str) == positive_label).astype(int)
    y_prob = pd.to_numeric(pred_df["churn_probability"], errors="coerce").fillna(0.0)

    accuracy_val = accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, zero_division=0)
    roc_auc_val = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy_val:.2%}")
    m2.metric("ROC-AUC", "N/A" if np.isnan(roc_auc_val) else f"{roc_auc_val:.4f}")
    m3.metric("F1-Score", f"{f1_val:.4f}")

    divider()
    section_header("Feature Importance", "Top Predictors")
    importance_df = predictor.get_feature_importance()
    if importance_df is not None and not importance_df.empty:
        top = importance_df.head(12).sort_values("importance")
        fig_importance = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Importance Scores",
            color="importance",
            color_continuous_scale=["#E0F2FE", "#0EA5E9", "#0284C7"],
        )
        fig_importance.update_coloraxes(showscale=False)
        st.plotly_chart(apply_chart_theme(fig_importance), width="stretch")
        st.dataframe(importance_df.head(20), width="stretch", height=320)
    else:
        st.info("Feature importance not available for this model type.")

    divider()
    section_header("Confusion Matrix & ROC", "Evaluation Curves")
    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted No Churn", "Predicted Churn"],
            y=["Actual No Churn", "Actual Churn"],
            colorscale=[[0, "#F0FDFF"], [1, "#0EA5E9"]],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(apply_chart_theme(fig_cm), width="stretch")

    with c2:
        if y_true.nunique() > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve", line=dict(color="#0EA5E9", width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(color="#9CA3AF", dash="dash")))
            fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc_val:.4f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(apply_chart_theme(fig_roc), width="stretch")
        else:
            st.info("ROC curve unavailable because only one class is present.")


# ─────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────

def page_about():
    page_hero(
        title="About ChurnSight",
        subtitle="Built to help businesses proactively retain customers through data-driven predictions.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### What is ChurnSight?

        ChurnSight is an end-to-end **Customer Churn Prediction Platform** that combines
        machine learning with an intuitive dashboard interface. It helps customer success
        and product teams identify at-risk customers early — and take action before it's too late.

        ### How it works

        1. **Data ingestion** — Upload customer data or use the real-time form
        2. **Preprocessing** — Automated feature engineering and encoding
        3. **Prediction** — XGBoost model scores each customer
        4. **Insights** — Risk levels, probability scores, and recommended actions are surfaced

        ### Tech stack
        """)
        st.markdown("""
        | Component | Technology |
        |-----------|-----------|
        | Framework | Streamlit |
        | ML Model | XGBoost / Scikit-learn |
        | Visualization | Plotly |
        | Data processing | Pandas, NumPy |
        | UI Design | Custom CSS + Manrope |
        """)

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>📌 Version Info</h3>
                <p><strong>Version:</strong> 2.0.0</p>
                <p><strong>Model:</strong> XGBoost v2.0</p>
                <p><strong>Last Updated:</strong> March 2026</p>
                <p><strong>Dataset:</strong> Telco Churn Dataset</p>
            </div>
            <div class="card">
                <h3>📬 Support</h3>
                <p>For questions or issues, please open a GitHub issue at <a href="https://github.com/rishab0003" target="_blank">https://github.com/rishab0003</a> or contact the ML team.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
#  MAIN APP ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    inject_global_css()

    page = render_sidebar()

    if page == "Home":
        page_home()
    elif page == "Predict Churn":
        page_predict()
    elif page == "Analytics":
        page_analytics()
    elif page == "Model Info":
        page_model_info()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
