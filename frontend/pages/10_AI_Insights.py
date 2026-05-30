"""
AI insights dashboard.
"""

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

st.markdown("""
Generate SHAP explainability and
AI-powered business insights.
""")

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
            response[
                "explainability"
            ]
        )

        importance_df = (
            pd.DataFrame(
                explainability[
                    "feature_importance"
                ]
            )
        )

        st.divider()

        st.subheader(
            "🔥 SHAP Feature Importance"
        )

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

        st.divider()

        st.subheader(
            "🤖 AI Business Insights"
        )

        insights = (
            response[
                "ai_insights"
            ]
        )

        st.markdown(
            insights
        )

        

    else:

        st.error(
            response.get(
                "detail",
                "AI insights failed."
            )
        )

        st.json(response)