"""
Data cleaning dashboard.
"""

import streamlit as st
import pandas as pd
import os
import requests

from utils.helpers import (
    set_page_config
)

from utils.api_client import (
    APIClient
)

set_page_config(
    "Data Cleaning"
)

st.title(
    "🧹 Automated Data Cleaning"
)

st.markdown("""
Clean datasets automatically using
advanced preprocessing techniques.
""")

st.divider()

st.subheader(
    "⚙ Cleaning Configuration"
)

missing_strategy = st.selectbox(
    "Missing Value Strategy",
    ["mean", "median"]
)

scaling_method = st.selectbox(
    "Scaling Method",
    ["standard", "minmax"]
)

if st.button(
    "🚀 Start Data Cleaning"
):

    with st.spinner(
        "Cleaning dataset..."
    ):

        response = (
            APIClient.clean_dataset(
                missing_strategy,
                scaling_method
            )
        )

    if response["status"] == "success":

        st.success(
            "Dataset cleaned successfully."
        )

        report = response[
            "cleaning_report"
        ]

        st.divider()

        st.subheader(
            "📊 Before vs After"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Before Cleaning"
            )

            st.metric(
                "Rows",
                report["before_shape"][0]
            )

            st.metric(
                "Columns",
                report["before_shape"][1]
            )

            st.metric(
                "Missing Values",
                report["before_missing"]
            )

            st.metric(
                "Duplicates",
                report[
                    "before_duplicates"
                ]
            )

        with col2:

            st.markdown(
                "### After Cleaning"
            )

            st.metric(
                "Rows",
                report["after_shape"][0]
            )

            st.metric(
                "Columns",
                report["after_shape"][1]
            )

            st.metric(
                "Missing Values",
                report["after_missing"]
            )

            st.metric(
                "Duplicates",
                report[
                    "after_duplicates"
                ]
            )

        st.divider()

        st.subheader(
            "🚨 Outlier Analysis"
        )

        # Create safe outlier comparison dataframe

        before_outliers = report[
            "outliers_before"
        ]

        after_outliers = report[
            "outliers_after"
        ]

        all_columns = list(
            set(before_outliers.keys())
            | set(after_outliers.keys())
        )

        outlier_data = []

        for column in all_columns:

            outlier_data.append(
                {
                    "Column": column,

                    "Outliers Before":
                        before_outliers.get(
                            column,
                            0
                        ),

                    "Outliers After":
                        after_outliers.get(
                            column,
                            0
                        ),
                }
            )

        outlier_df = pd.DataFrame(
            outlier_data
        )

        st.dataframe(
            outlier_df,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "⬇ Download Cleaned Dataset"
        )

        backend_url = os.getenv(
            "BACKEND_URL",
            "http://127.0.0.1:8000"
        )

        download_url = (
            f"{backend_url}"
            "/api/download-cleaned-data"
        )

        download_response = requests.get(
            download_url
        )

        if download_response.status_code == 200:

            st.download_button(
                label="Download Cleaned CSV",
                data=download_response.content,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

        else:

            st.error(
                "Failed to download cleaned dataset."
            )

    else:

        st.error(
            response.get(
                "message",
                "Cleaning failed."
            )
        )