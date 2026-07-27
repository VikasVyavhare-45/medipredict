"""
Stroke Risk Screening page.
Thin wrapper - the actual form is generated from models/stroke/scaler.pkl
by form_builder.py, so it can never fall out of sync with the trained model.
"""

import sys, os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from form_builder import render_disease_form

st.set_page_config(page_title="Stroke Risk Screening", page_icon="🧬", layout="wide")

render_disease_form("stroke")
