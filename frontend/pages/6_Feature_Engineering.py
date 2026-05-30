"""
Feature engineering dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.helpers import (
    set_page_config
)

from utils.api_client import (
    APIClient
)

set_page_config(
    "Feature Engineering"
)

st.title(
    "🧠 Feature Engineering"
)

summary_response = (
    APIClient.get_dataset_summary()
)

if (
    summary_response["status"]
    != "success"
):

    st.error(
        "Upload dataset first."
    )

    st.stop()

summary = (
    summary_response["summary"]
)

columns = (
    summary["column_names"]
)

st.divider()

st.subheader(
    "⚙ Configuration"
)

target_column = st.selectbox(
    "Target Column",
    columns
)

correlation_threshold = (
    st.slider(
        "Correlation Threshold",
        0.5,
        1.0,
        0.9
    )
)

variance_threshold = (
    st.slider(
        "Variance Threshold",
        0.0,
        1.0,
        0.01
    )
)

pca_components = (
    st.slider(
        "PCA Components",
        2,
        10,
        2
    )
)

if st.button(
    "🚀 Run Feature Engineering"
):

    with st.spinner(
        "Processing..."
    ):

        response = (
            APIClient
            .feature_engineering(
                target_column,
                correlation_threshold,
                variance_threshold,
                pca_components
            )
        )

    if response.get("status") == "success":

        st.success(
            "Feature engineering completed."
        )

        st.divider()

        st.subheader(
            "🗑 Removed Correlated Features"
        )

        st.write(
            response[
                "removed_correlated_features"
            ]
        )

        st.divider()

        st.subheader(
            "⚠ Removed Low Variance Features"
        )

        st.write(
            response[
                "removed_low_variance_features"
            ]
        )

        st.divider()

        st.subheader(
            "📉 PCA Explained Variance"
        )

        pca_df = pd.DataFrame(
            {
                "Component": [
                    f"PC{i+1}"
                    for i in range(
                        len(
                            response[
                                "pca_explained_variance"
                            ]
                        )
                    )
                ],

                "Explained Variance":
                    response[
                        "pca_explained_variance"
                    ]
            }
        )

        pca_chart = px.bar(
            pca_df,
            x="Component",
            y="Explained Variance",
            title="PCA Explained Variance"
        )

        st.plotly_chart(
            pca_chart,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "📊 Train/Test Split"
        )

        split_info = (
            response[
                "train_test_split"
            ]
        )

        x_train_shape = (
            split_info["x_train_shape"]
        )

        x_test_shape = (
            split_info["x_test_shape"]
        )

        y_train_shape = (
            split_info["y_train_shape"]
        )

        y_test_shape = (
            split_info["y_test_shape"]
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Train Rows",
            x_train_shape[0]
        )

        col2.metric(
            "Test Rows",
            x_test_shape[0]
        )

        col3.metric(
            "Train Features",
            x_train_shape[1]
        )

        col4.metric(
            "Test Features",
            x_test_shape[1]
        )

        st.write(
            f"X Train Shape: "
            f"{tuple(x_train_shape)}"
        )

        st.write(
            f"X Test Shape: "
            f"{tuple(x_test_shape)}"
        )

        st.write(
            f"Y Train Shape: "
            f"{tuple(y_train_shape)}"
        )

        st.write(
            f"Y Test Shape: "
            f"{tuple(y_test_shape)}"
        )

        st.divider()

        st.subheader(
            "🔥 Feature Importance"
        )

        importance_df = (
            pd.DataFrame(
                response[
                    "feature_importance"
                ]
            )
        )

        importance_chart = px.bar(
            importance_df.head(20),
            x="Feature",
            y="Importance",
            title=(
                "Top Feature Importance"
            )
        )

        st.plotly_chart(
            importance_chart,
            use_container_width=True
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "⬇ Download Dataset"
        )

        download_response = (
            APIClient
            .download_transformed_dataset()
        )

        if (
            download_response
            .status_code == 200
        ):

            st.download_button(
                label=(
                    "Download "
                    "Transformed Dataset"
                ),
                data=download_response.content,
                file_name=(
                    "transformed_dataset.csv"
                ),
                mime="text/csv"
            )

        else:

            st.error(
                "Failed to download dataset."
            )

    else:

        st.error(
            response.get(
                "detail",
                response.get(
                    "message",
                    "Feature engineering failed."
                )
            )
        )

        st.json(response)