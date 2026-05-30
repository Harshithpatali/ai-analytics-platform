"""
Data cleaning dashboard.
"""

import os

import pandas as pd
import requests
import streamlit as st

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

st.markdown(
    """
Clean datasets automatically using
advanced preprocessing techniques.
"""
)

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

    if response.get("status") == "success":

        st.success(
            "Dataset cleaned successfully."
        )

        report = response.get(
            "cleaning_report",
            {}
        )

        st.divider()

        st.subheader(
            "📊 Before vs After"
        )

        col1, col2 = st.columns(2)

        before_shape = report.get(
            "before_shape",
            [0, 0]
        )

        after_shape = report.get(
            "after_shape",
            [0, 0]
        )

        with col1:

            st.markdown(
                "### Before Cleaning"
            )

            st.metric(
                "Rows",
                before_shape[0]
            )

            st.metric(
                "Columns",
                before_shape[1]
            )

            st.metric(
                "Missing Values",
                report.get(
                    "before_missing",
                    0
                )
            )

            st.metric(
                "Duplicates",
                report.get(
                    "before_duplicates",
                    0
                )
            )

        with col2:

            st.markdown(
                "### After Cleaning"
            )

            st.metric(
                "Rows",
                after_shape[0]
            )

            st.metric(
                "Columns",
                after_shape[1]
            )

            st.metric(
                "Missing Values",
                report.get(
                    "after_missing",
                    0
                )
            )

            st.metric(
                "Duplicates",
                report.get(
                    "after_duplicates",
                    0
                )
            )

        st.divider()

        st.subheader(
            "🚨 Outlier Analysis"
        )

        before_outliers = report.get(
            "outliers_before",
            {}
        )

        after_outliers = report.get(
            "outliers_after",
            {}
        )

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
            "https://ai-analytics-platform-rgi6.onrender.com"
        )

        download_url = (
            f"{backend_url}"
            "/api/download-cleaned-data"
        )

        try:

            download_response = requests.get(
                download_url,
                timeout=60
            )

            if (
                download_response.status_code
                == 200
            ):

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

        except Exception as error:

            st.error(
                f"Download error: {error}"
            )

    else:

        st.error(
            response.get(
                "message",
                "Cleaning failed."
            )
        )