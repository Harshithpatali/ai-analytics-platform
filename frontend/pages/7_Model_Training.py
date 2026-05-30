"""
Model training dashboard.
"""

import traceback
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.helpers import (
    set_page_config
)

from utils.api_client import (
    APIClient
)

set_page_config(
    "Model Training"
)

st.title(
    "🤖 Model Training"
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

target_column = st.selectbox(
    "Target Column",
    summary["column_names"]
)

if st.button(
    "🚀 Train Models"
):

    try:

        with st.spinner(
            "Training machine learning models..."
        ):

            response = (
                APIClient.train_models(
                    target_column
                )
            )

        if (
            response.get("status")
            == "success"
        ):

            training_results = (
                response[
                    "training_results"
                ]
            )

            problem_type = (
                training_results[
                    "problem_type"
                ]
            )

            results = (
                training_results[
                    "results"
                ]
            )

            best_model = (
                training_results[
                    "best_model"
                ]
            )

            recommendation = (
                training_results[
                    "recommendation"
                ]
            )

            st.success(
                "Models trained successfully."
            )

            st.divider()

            st.subheader(
                "🏆 Best Model"
            )

            st.success(
                best_model
            )

            st.info(
                f"Recommended Model: "
                f"{recommendation['recommended_model']}"
            )

            st.write(
                f"Score: "
                f"{recommendation['score']:.4f}"
            )

            metrics_data = []

            for model_name, result in (
                results.items()
            ):

                row = {
                    "Model":
                        model_name
                }

                row.update(
                    result["metrics"]
                )

                metrics_data.append(
                    row
                )

            metrics_df = pd.DataFrame(
                metrics_data
            )

            st.divider()

            st.subheader(
                "📊 Model Metrics"
            )

            st.dataframe(
                metrics_df,
                use_container_width=True
            )

            # Keep only numeric metric columns

            numeric_columns = []

            for column in metrics_df.columns:

                if column == "Model":
                    continue

                first_value = (
                    metrics_df[column]
                    .dropna()
                    .iloc[0]
                    if not metrics_df[column]
                    .dropna()
                    .empty
                    else None
                )

                if isinstance(
                    first_value,
                    (int, float)
                ):

                    numeric_columns.append(
                        column
                    )

            for metric in numeric_columns:

                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y=metric,
                    title=f"{metric} Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.divider()

            st.subheader(
                "📌 Individual Model Results"
            )

            for model_name, result in (
                results.items()
            ):

                with st.expander(
                    model_name,
                    expanded=False
                ):

                    st.write(
                        "### Metrics"
                    )

                    st.json(
                        result["metrics"]
                    )

                    st.write(
                        "### Analytics"
                    )

                    st.json(
                        result["analytics"]
                    )

                    st.write(
                        "### Model Path"
                    )

                    st.code(
                        result["model_path"]
                    )

        else:

            st.error(
                response.get(
                    "detail",
                    "Training failed."
                )
            )

            st.json(response)

    except Exception as error:

        st.error(
            "Unexpected error occurred."
        )

        st.code(
            traceback.format_exc()
        )