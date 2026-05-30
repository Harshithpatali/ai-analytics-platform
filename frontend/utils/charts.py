"""
Reusable chart utilities.
"""

import plotly.express as px
import pandas as pd


def create_missing_values_chart(df: pd.DataFrame):
    """
    Create missing values bar chart.
    """

    missing_data = df.isnull().sum().reset_index()

    missing_data.columns = ["Column", "Missing Values"]

    fig = px.bar(
        missing_data,
        x="Column",
        y="Missing Values",
        title="Missing Values Analysis"
    )

    return fig