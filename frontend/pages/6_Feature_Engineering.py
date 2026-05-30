"""
Feature engineering dashboard.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

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
    summary_response.get("status")
    != "success"
):

    st.error(
        summary_response.get(
            "message",
            "Upload dataset first."
        )
    )

    st.stop()

summary = (
    summary_response.get(
        "summary",
        {}
    )
)

columns = (
    summary.get(
        "column_names",
        []
    )
)

if not columns:

    st.warning(
        "No columns found in dataset."
    )

    st.stop()

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
            response.get(
                "removed_correlated_features",
                []
            )
        )

        st.divider()

        st.subheader(
            "⚠ Removed Low Variance Features"
        )

        st.write(
            response.get(
                "removed_low_variance_features",
                []
            )
        )

        st.divider()

        st.subheader(
            "📉 PCA Explained Variance"
        )

        pca_variance = response.get(
            "pca_explained_variance",
            []
        )

        if pca_variance:

            pca_df = pd.DataFrame(
                {
                    "Component": [
                        f"PC{i+1}"
                        for i in range(
                            len(
                                pca_variance
                            )
                        )
                    ],

                    "Explained Variance":
                        pca_variance
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

        else:

            st.warning(
                "No PCA variance data found."
            )

        st.divider()

        st.subheader(
            "📊 Train/Test Split"
        )

        split_info = (
            response.get(
                "train_test_split",
                {}
            )
        )

        x_train_shape = (
            split_info.get(
                "x_train_shape",
                [0, 0]
            )
        )

        x_test_shape = (
            split_info.get(
                "x_test_shape",
                [0, 0]
            )
        )

        y_train_shape = (
            split_info.get(
                "y_train_shape",
                [0]
            )
        )

        y_test_shape = (
            split_info.get(
                "y_test_shape",
                [0]
            )
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

        feature_importance = response.get(
            "feature_importance",
            []
        )

        if feature_importance:

            importance_df = (
                pd.DataFrame(
                    feature_importance
                )
            )

            if (
                "Feature"
                in importance_df.columns
                and
                "Importance"
                in importance_df.columns
            ):

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

        else:

            st.warning(
                "No feature importance data found."
            )

        st.divider()

        st.subheader(
            "⬇ Download Dataset"
        )

        try:

            download_response = (
                APIClient
                .download_transformed_dataset()
            )

            if (
                download_response
                and
                download_response.status_code
                == 200
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

        except Exception as error:

            st.error(
                f"Download error: {error}"
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