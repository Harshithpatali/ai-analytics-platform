"""
AI insights dashboard.
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
    "AI Insights"
)

st.title(
    "🧠 Explainable AI & Business Insights"
)

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "SVM",
    "Linear Regression",
    "Gradient Boosting"
]

model_name = st.selectbox(
    "Select Model",
    models
)

st.divider()

st.markdown(
    """
Generate SHAP explainability and
AI-powered business insights.
"""
)

if st.button(
    "Generate AI Insights"
):

    with st.spinner(
        "Generating insights..."
    ):

        response = (
            APIClient.predict(
                model_name
            )
        )

    if (
        response.get("status")
        == "success"
    ):

        st.success(
            "AI insights generated."
        )

        explainability = (
            response.get(
                "explainability",
                {}
            )
        )

        feature_importance = (
            explainability.get(
                "feature_importance",
                []
            )
        )

        st.divider()

        st.subheader(
            "🔥 SHAP Feature Importance"
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

                fig = px.bar(
                    importance_df.head(20),
                    x="Feature",
                    y="Importance",
                    title=(
                        "Global Feature Importance"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.divider()

            st.subheader(
                "📊 Feature Importance Table"
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
            "🤖 AI Business Insights"
        )

        insights = (
            response.get(
                "ai_insights",
                "No AI insights available."
            )
        )

        st.markdown(
            insights
        )

    else:

        st.error(
            response.get(
                "detail",
                response.get(
                    "message",
                    "AI insights failed."
                )
            )
        )

        st.json(response)