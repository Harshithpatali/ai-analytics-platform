"""
Dataset upload page.
"""

import streamlit as st
import pandas as pd

from utils.helpers import (
    set_page_config
)

from utils.api_client import APIClient


set_page_config("Upload Dataset")

st.title("📂 Upload Dataset")

st.markdown("""
Upload CSV or Excel datasets for
automated analytics and machine learning.
""")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file:

    st.info(
        f"Selected File: {uploaded_file.name}"
    )

    with st.spinner(
        "Uploading dataset..."
    ):

        response = (
            APIClient
            .upload_dataset(uploaded_file)
        )

    if response["status"] == "success":

        st.success(
            "Dataset uploaded successfully."
        )

        summary = response["summary"]

        st.subheader(
            "📊 Dataset Summary"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            summary["rows"]
        )

        col2.metric(
            "Columns",
            summary["columns"]
        )

        col3.metric(
            "Missing Values",
            sum(
                summary[
                    "missing_values"
                ].values()
            )
        )

        col4.metric(
            "Duplicates",
            summary["duplicate_rows"]
        )

        st.divider()

        st.subheader(
            "🧠 Dataset Metadata"
        )

        st.write(
            f"Memory Usage: "
            f"{summary['memory_usage_mb']} MB"
        )

        st.write(
            "Numerical Columns:",
            summary["numerical_columns"]
        )

        st.write(
            "Categorical Columns:",
            summary["categorical_columns"]
        )

        st.divider()

        st.subheader(
            "📌 Column Data Types"
        )

        dtype_df = pd.DataFrame(
            {
                "Column": summary[
                    "data_types"
                ].keys(),
                "Data Type": summary[
                    "data_types"
                ].values(),
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "⚠ Missing Values"
        )

        missing_df = pd.DataFrame(
            {
                "Column": summary[
                    "missing_values"
                ].keys(),
                "Missing Values": summary[
                    "missing_values"
                ].values(),
            }
        )

        st.dataframe(
            missing_df,
            use_container_width=True
        )

    else:

        st.error(
            response.get(
                "message",
                "Upload failed."
            )
        )