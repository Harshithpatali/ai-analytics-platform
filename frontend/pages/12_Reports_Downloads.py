"""
Reports and downloads dashboard.
"""

import streamlit as st

from utils.helpers import (
    set_page_config
)

set_page_config(
    "Reports & Downloads"
)

st.title(
    "📄 Reports & Downloads"
)

st.markdown(
    """
Download generated reports,
datasets, predictions, and analytics outputs.
"""
)

st.divider()

st.info(
    "Reports module coming soon."
)

st.subheader(
    "Available Downloads"
)

downloads = [
    "Cleaned Dataset",
    "Predictions CSV",
    "Feature Engineering Output",
    "Model Metrics Report",
    "AI Insights Report"
]

for item in downloads:

    st.write(
        f"• {item}"
    )