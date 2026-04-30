import streamlit as st
from src.ui import render_tab1, render_tab2

st.set_page_config(
    page_title="Human Detection Inference",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Add custom CSS for a premium feel
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Human Detection Dashboard")

tab1, tab2 = st.tabs(["📊 Model Metrics", "🎥 Video Inference"])

with tab1:
    render_tab1()
    
with tab2:
    render_tab2()
