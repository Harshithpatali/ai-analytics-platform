"""
EDA dashboard.
"""

import json
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
    summary_response["status"]
    != "success"
):

    st.error(
        "Failed to load EDA summary."
    )

    st.stop()

summary = (
    summary_response[
        "eda_summary"
    ]
)

numerical_columns = (
    summary[
        "numerical_columns"
    ]
)

categorical_columns = (
    summary[
        "categorical_columns"
    ]
)

st.divider()

st.subheader(
    "📌 Dataset Summary"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    summary["shape"][0]
)

col2.metric(
    "Columns",
    summary["shape"][1]
)

col3.metric(
    "Duplicates",
    summary["duplicate_rows"]
)

st.divider()

st.subheader(
    "📈 Numerical Analysis"
)

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

fig = pio.from_json(
    chart_response["chart"]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

st.subheader(
    "🔗 Scatter Plot Analysis"
)

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

scatter_fig = pio.from_json(
    scatter_response["chart"]
)

st.plotly_chart(
    scatter_fig,
    use_container_width=True
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

heatmap_fig = pio.from_json(
    heatmap_response["chart"]
)

st.plotly_chart(
    heatmap_fig,
    use_container_width=True
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

missing_fig = pio.from_json(
    missing_response["chart"]
)

st.plotly_chart(
    missing_fig,
    use_container_width=True
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

    cat_fig = pio.from_json(
        cat_response["chart"]
    )

    st.plotly_chart(
        cat_fig,
        use_container_width=True
    )

else:

    st.info(
        "No categorical columns found."
    )