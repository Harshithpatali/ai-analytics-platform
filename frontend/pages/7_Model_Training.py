"""
Model training dashboard.
"""

import traceback

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
    "Model Training"
)

st.title(
    "🤖 Model Training"
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

column_names = (
    summary.get(
        "column_names",
        []
    )
)

if not column_names:

    st.warning(
        "No dataset columns found."
    )

    st.stop()

target_column = st.selectbox(
    "Target Column",
    column_names
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
                response.get(
                    "training_results",
                    {}
                )
            )

            problem_type = (
                training_results.get(
                    "problem_type",
                    "Unknown"
                )
            )

            results = (
                training_results.get(
                    "results",
                    {}
                )
            )

            best_model = (
                training_results.get(
                    "best_model",
                    "Unknown"
                )
            )

            recommendation = (
                training_results.get(
                    "recommendation",
                    {}
                )
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

            st.write(
                f"Problem Type: "
                f"{problem_type}"
            )

            st.info(
                f"Recommended Model: "
                f"{recommendation.get('recommended_model', 'N/A')}"
            )

            score = recommendation.get(
                "score"
            )

            if score is not None:

                st.write(
                    f"Score: "
                    f"{score:.4f}"
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
                    result.get(
                        "metrics",
                        {}
                    )
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

            if not metrics_df.empty:

                st.dataframe(
                    metrics_df,
                    use_container_width=True
                )

                numeric_columns = []

                for column in metrics_df.columns:

                    if column == "Model":

                        continue

                    clean_series = (
                        metrics_df[column]
                        .dropna()
                    )

                    if clean_series.empty:

                        continue

                    first_value = (
                        clean_series.iloc[0]
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

            else:

                st.warning(
                    "No model metrics found."
                )

            st.divider()

            st.subheader(
                "📌 Individual Model Results"
            )

            if results:

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
                            result.get(
                                "metrics",
                                {}
                            )
                        )

                        st.write(
                            "### Analytics"
                        )

                        st.json(
                            result.get(
                                "analytics",
                                {}
                            )
                        )

                        st.write(
                            "### Model Path"
                        )

                        st.code(
                            result.get(
                                "model_path",
                                "N/A"
                            )
                        )

            else:

                st.warning(
                    "No training results available."
                )

        else:

            st.error(
                response.get(
                    "detail",
                    response.get(
                        "message",
                        "Training failed."
                    )
                )
            )

            st.json(response)

    except Exception:

        st.error(
            "Unexpected error occurred."
        )

        st.code(
            traceback.format_exc()
        )