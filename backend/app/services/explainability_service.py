"""
Explainability service.
"""

import pandas as pd
import numpy as np


class ExplainabilityService:
    """
    Handles model explainability.
    """

    @staticmethod
    def generate_feature_importance(
        model,
        feature_columns
    ):
        """
        Generate feature importance safely.
        """

        importance_scores = None

        # Tree models
        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_scores = (
                model.feature_importances_
            )

        # Linear models
        elif hasattr(
            model,
            "coef_"
        ):

            importance_scores = (
                np.abs(
                    model.coef_
                )
            )

            if len(
                importance_scores.shape
            ) > 1:

                importance_scores = (
                    importance_scores[0]
                )

        else:

            importance_scores = (
                [0]
                * len(feature_columns)
            )

        # Convert safely to lists
        importance_scores = list(
            importance_scores
        )

        feature_columns = list(
            feature_columns
        )

        # Align lengths safely
        min_length = min(
            len(feature_columns),
            len(importance_scores)
        )

        aligned_features = (
            feature_columns[:min_length]
        )

        aligned_scores = (
            importance_scores[:min_length]
        )

        feature_importance = (
            pd.DataFrame(
                {
                    "Feature":
                        aligned_features,

                    "Importance":
                        aligned_scores
                }
            )
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        return {
            "feature_importance":
                feature_importance
                .to_dict(
                    orient="records"
                )
        }