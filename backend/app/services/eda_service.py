"""
EDA service for automated analysis.
"""

import pandas as pd
from typing import Dict, Any


class EDAService:
    """
    Handles exploratory data analysis.
    """

    @staticmethod
    def generate_eda_summary(
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate EDA summary.
        """

        numerical_columns = list(
            df.select_dtypes(
                include=["int64", "float64"]
            ).columns
        )

        categorical_columns = list(
            df.select_dtypes(
                include=["object", "category"]
            ).columns
        )

        summary = {
            "shape": df.shape,
            "numerical_columns":
                numerical_columns,

            "categorical_columns":
                categorical_columns,

            "missing_values": (
                df.isnull()
                .sum()
                .to_dict()
            ),

            "duplicate_rows":
                int(df.duplicated().sum()),

            "descriptive_statistics":
                df.describe()
                .to_dict(),

            "correlation_matrix":
                df.select_dtypes(
                    include=[
                        "int64",
                        "float64"
                    ]
                )
                .corr()
                .round(2)
                .to_dict()
        }

        return summary