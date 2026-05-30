"""
Visualization service.
"""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


class VisualizationService:
    """
    Handles Plotly chart generation.
    """

    @staticmethod
    def histogram(
        df: pd.DataFrame,
        column: str
    ):
        """
        Generate histogram.
        """

        fig = px.histogram(
            df,
            x=column,
            title=f"Histogram of {column}"
        )

        return fig.to_json()

    @staticmethod
    def boxplot(
        df: pd.DataFrame,
        column: str
    ):
        """
        Generate boxplot.
        """

        fig = px.box(
            df,
            y=column,
            title=f"Boxplot of {column}"
        )

        return fig.to_json()

    @staticmethod
    def scatter_plot(
        df: pd.DataFrame,
        x_column: str,
        y_column: str
    ):
        """
        Generate scatter plot.
        """

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=(
                f"{x_column} vs "
                f"{y_column}"
            )
        )

        return fig.to_json()

    @staticmethod
    def correlation_heatmap(
        df: pd.DataFrame
    ):
        """
        Generate correlation heatmap.
        """

        correlation = (
            df.select_dtypes(
                include=[
                    "int64",
                    "float64"
                ]
            )
            .corr()
            .round(2)
        )

        fig = ff.create_annotated_heatmap(
            z=correlation.values,
            x=list(correlation.columns),
            y=list(correlation.index),
            annotation_text=(
                correlation.values
                .round(2)
                .astype(str)
            ),
            showscale=True
        )

        return fig.to_json()

    @staticmethod
    def missing_value_heatmap(
        df: pd.DataFrame
    ):
        """
        Generate missing value heatmap.
        """

        missing_df = (
            df.isnull()
            .astype(int)
        )

        fig = px.imshow(
            missing_df,
            aspect="auto",
            title="Missing Value Heatmap"
        )

        return fig.to_json()

    @staticmethod
    def categorical_distribution(
        df: pd.DataFrame,
        column: str
    ):
        """
        Generate categorical chart.
        """

        value_counts = (
            df[column]
            .value_counts()
            .reset_index()
        )

        value_counts.columns = [
            "Category",
            "Count"
        ]

        fig = px.bar(
            value_counts,
            x="Category",
            y="Count",
            title=(
                f"Distribution of "
                f"{column}"
            )
        )

        return fig.to_json()