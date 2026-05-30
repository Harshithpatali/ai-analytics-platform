"""
Home dashboard page.
"""

import streamlit as st

from utils.helpers import set_page_config


set_page_config("Dashboard")

st.title("📊 Dashboard")

st.markdown("## Platform Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Datasets", "0")
col2.metric("Models", "0")
col3.metric("Predictions", "0")
col4.metric("Reports", "0")

st.divider()

st.subheader("🚀 Workflow")

workflow_steps = [
    "1. Upload Dataset",
    "2. Clean Data",
    "3. Perform EDA",
    "4. Train Models",
    "5. Generate Predictions",
    "6. Get AI Insights"
]

for step in workflow_steps:
    st.write(step)