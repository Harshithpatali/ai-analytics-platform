"""
EDA dashboard.
"""

import streamlit as st
import plotly.io as pio

from utils.helpers import (
    set_page_config
)

from utils.api_client import (
    APIClient
)

set_page_config(
    "EDA Dashboard"
)

st.title(
    "📊 Exploratory Data Analysis"
)

summary_response = (
    APIClient.get_eda_summary()
)

if (
    summary_response.get("status")
    != "success"
):

    st.error(
        summary_response.get(
            "message",
            "Failed to load EDA summary."
        )
    )

    st.stop()

summary = (
    summary_response.get(
        "eda_summary",
        {}
    )
)

numerical_columns = (
    summary.get(
        "numerical_columns",
        []
    )
)

categorical_columns = (
    summary.get(
        "categorical_columns",
        []
    )
)

shape = summary.get(
    "shape",
    [0, 0]
)

st.divider()

st.subheader(
    "📌 Dataset Summary"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    shape[0]
)

col2.metric(
    "Columns",
    shape[1]
)

col3.metric(
    "Duplicates",
    summary.get(
        "duplicate_rows",
        0
    )
)

st.divider()

st.subheader(
    "📈 Numerical Analysis"
)

if numerical_columns:

    selected_num_column = (
        st.selectbox(
            "Select Numerical Column",
            numerical_columns
        )
    )

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Histogram",
            "Boxplot"
        ]
    )

    if chart_type == "Histogram":

        chart_response = (
            APIClient.get_chart(
                "histogram",
                {
                    "column":
                    selected_num_column
                }
            )
        )

    else:

        chart_response = (
            APIClient.get_chart(
                "boxplot",
                {
                    "column":
                    selected_num_column
                }
            )
        )

    chart_json = chart_response.get(
        "chart"
    )

    if chart_json:

        fig = pio.from_json(
            chart_json
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Failed to generate chart."
        )

else:

    st.info(
        "No numerical columns found."
    )

st.divider()

st.subheader(
    "🔗 Scatter Plot Analysis"
)

if len(numerical_columns) >= 2:

    x_column = st.selectbox(
        "X-Axis",
        numerical_columns,
        key="x"
    )

    y_column = st.selectbox(
        "Y-Axis",
        numerical_columns,
        key="y"
    )

    scatter_response = (
        APIClient.get_chart(
            "scatter-plot",
            {
                "x_column":
                x_column,

                "y_column":
                y_column
            }
        )
    )

    scatter_chart = (
        scatter_response.get(
            "chart"
        )
    )

    if scatter_chart:

        scatter_fig = pio.from_json(
            scatter_chart
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Failed to generate scatter plot."
        )

else:

    st.info(
        "At least two numerical columns are required."
    )

st.divider()

st.subheader(
    "🔥 Correlation Heatmap"
)

heatmap_response = (
    APIClient.get_chart(
        "correlation-heatmap"
    )
)

heatmap_chart = (
    heatmap_response.get(
        "chart"
    )
)

if heatmap_chart:

    heatmap_fig = pio.from_json(
        heatmap_chart
    )

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )

else:

    st.warning(
        "Failed to generate correlation heatmap."
    )

st.divider()

st.subheader(
    "⚠ Missing Value Heatmap"
)

missing_response = (
    APIClient.get_chart(
        "missing-value-heatmap"
    )
)

missing_chart = (
    missing_response.get(
        "chart"
    )
)

if missing_chart:

    missing_fig = pio.from_json(
        missing_chart
    )

    st.plotly_chart(
        missing_fig,
        use_container_width=True
    )

else:

    st.warning(
        "Failed to generate missing value heatmap."
    )

st.divider()

st.subheader(
    "📦 Categorical Analysis"
)

if categorical_columns:

    selected_cat_column = (
        st.selectbox(
            "Select Categorical Column",
            categorical_columns
        )
    )

    cat_response = (
        APIClient.get_chart(
            "categorical-distribution",
            {
                "column":
                selected_cat_column
            }
        )
    )

    cat_chart = cat_response.get(
        "chart"
    )

    if cat_chart:

        cat_fig = pio.from_json(
            cat_chart
        )

        st.plotly_chart(
            cat_fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Failed to generate categorical chart."
        )

else:

    st.info(
        "No categorical columns found."
    )