import os
import pickle
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Obsidian Base with Cyan & Violet Neon Accents
st.markdown("""
<style>
    /* Global background and typography */
    .stApp {
        background: radial-gradient(circle at top left, #111827, #030712 70%);
        color: #f3f4f6;
    }

    /* Main Hero Header */
    .hero-header {
        text-align: center;
        padding: 2rem 0 2.5rem 0;
    }
    .hero-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .hero-header p {
        font-size: 1.05rem;
        color: #9ca3af;
        font-weight: 400;
    }

    /* Modern Card Layout with Glow Shadows */
    .custom-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 
                    0 0 15px -3px rgba(56, 189, 248, 0.08);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.8), 
                    0 0 25px rgba(56, 189, 248, 0.15);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        padding-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: 0.02em;
    }

    /* Streamlit Input Styling Adjustments */
    div[data-baseweb="input"] {
        background-color: #0b0f19 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #38bdf8 !important
