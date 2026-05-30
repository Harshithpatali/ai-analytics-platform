"""
Model comparison dashboard.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    set_page_config
)

from utils.api_client import (
    APIClient
)

set_page_config(
    "Model Comparison"
)

st.title(
    "🏆 Model Comparison Dashboard"
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
    "🚀 Compare Models"
):

    with st.spinner(
        "Training and comparing..."
    ):

        response = (
            APIClient.compare_models(
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

        recommendation = (
            training_results[
                "recommendation"
            ]
        )

        st.success(
            "Model comparison completed."
        )

        st.divider()

        st.subheader(
            "🏆 Recommended Model"
        )

        st.success(
            recommendation[
                "recommended_model"
            ]
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

            metrics_data.append(row)

        metrics_df = pd.DataFrame(
            metrics_data
        )

        st.divider()

        st.subheader(
            "📊 Metrics Comparison"
        )

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        numeric_columns = (
            metrics_df.select_dtypes(
                include=[
                    "float64",
                    "int64"
                ]
            ).columns
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

        for model_name, result in (
            results.items()
        ):

            st.subheader(
                f"📌 {model_name}"
            )

            analytics = (
                result[
                    "analytics"
                ]
            )

            if (
                problem_type
                == "classification"
            ):

                # Confusion Matrix
                if (
                    "confusion_matrix"
                    in analytics
                ):

                    cm = pd.DataFrame(
                        analytics[
                            "confusion_matrix"
                        ]
                    )

                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        title=(
                            f"{model_name} "
                            "Confusion Matrix"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                # ROC Curve
                if (
                    analytics.get(
                        "roc_curve"
                    )
                ):

                    roc_data = analytics[
                        "roc_curve"
                    ]

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=roc_data["fpr"],
                            y=roc_data["tpr"],
                            mode="lines",
                            name="ROC Curve"
                        )
                    )

                    fig.update_layout(
                        title=(
                            f"{model_name} "
                            "ROC Curve"
                        ),
                        xaxis_title="FPR",
                        yaxis_title="TPR"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                # Precision Recall Curve
                if (
                    analytics.get(
                        "precision_recall_curve"
                    )
                ):

                    pr_data = analytics[
                        "precision_recall_curve"
                    ]

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=pr_data["recall"],
                            y=pr_data[
                                "precision"
                            ],
                            mode="lines",
                            name="PR Curve"
                        )
                    )

                    fig.update_layout(
                        title=(
                            f"{model_name} "
                            "Precision Recall Curve"
                        ),
                        xaxis_title="Recall",
                        yaxis_title="Precision"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            else:

                actual_values = (
                    analytics[
                        "actual_values"
                    ]
                )

                predicted_values = (
                    analytics[
                        "predicted_values"
                    ]
                )

                residuals = (
                    analytics[
                        "residuals"
                    ]
                )

                # Actual vs Predicted
                fig = px.scatter(
                    x=actual_values,
                    y=predicted_values,
                    labels={
                        "x":
                            "Actual",

                        "y":
                            "Predicted"
                    },
                    title=(
                        f"{model_name} "
                        "Actual vs Predicted"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Residual Analysis
                residual_fig = px.histogram(
                    residuals,
                    title=(
                        f"{model_name} "
                        "Residual Distribution"
                    )
                )

                st.plotly_chart(
                    residual_fig,
                    use_container_width=True
                )

    else:

        st.error(
            response.get(
                "detail",
                "Comparison failed."
            )
        )

        st.json(response)