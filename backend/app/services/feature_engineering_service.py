"""
Feature engineering service.
"""

import os
import pandas as pd
import numpy as np

from typing import Dict, Any

from sklearn.decomposition import PCA

from sklearn.feature_selection import (
    VarianceThreshold
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

TRANSFORMED_DATA_DIR = (
    "app/reports"
)


class FeatureEngineeringService:
    """
    Handles feature engineering tasks.
    """

    @staticmethod
    def remove_correlated_features(
        df: pd.DataFrame,
        threshold: float = 0.9
    ):
        """
        Remove highly correlated features.
        """

        # Keep only numerical columns
        numerical_df = df.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )

        correlation_matrix = (
            numerical_df.corr().abs()
        )

        upper_triangle = (
            correlation_matrix.where(
                np.triu(
                    np.ones(
                        correlation_matrix.shape
                    ),
                    k=1
                ).astype(bool)
            )
        )

        columns_to_drop = [
            column
            for column in upper_triangle.columns
            if any(
                upper_triangle[column]
                > threshold
            )
        ]

        reduced_df = numerical_df.drop(
            columns=columns_to_drop
        )

        return (
            reduced_df,
            columns_to_drop
        )

    @staticmethod
    def variance_threshold_selection(
        df: pd.DataFrame,
        threshold: float = 0.01
    ):
        """
        Remove low variance features.
        """

        # Remove NaN values
        df = df.dropna()

        # Keep only numerical columns
        numerical_df = df.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )

        if numerical_df.empty:

            raise ValueError(
                "No numerical columns available "
                "for variance threshold."
            )

        selector = (
            VarianceThreshold(
                threshold=threshold
            )
        )

        transformed_data = (
            selector.fit_transform(
                numerical_df
            )
        )

        selected_columns = (
            numerical_df.columns[
                selector.get_support()
            ]
        )

        transformed_df = pd.DataFrame(
            transformed_data,
            columns=selected_columns
        )

        # Force numeric dtype
        transformed_df = (
            transformed_df.astype(float)
        )

        removed_columns = [
            column
            for column in numerical_df.columns
            if column not in selected_columns
        ]

        return (
            transformed_df,
            removed_columns
        )

    @staticmethod
    def apply_pca(
        df: pd.DataFrame,
        n_components: int = 2
    ):
        """
        Apply PCA transformation.
        """

        # Remove remaining NaN values
        df = df.dropna()

        # Keep only numerical columns
        numerical_df = df.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )

        if numerical_df.empty:

            raise ValueError(
                "No numerical columns available "
                "for PCA."
            )

        if (
            n_components
            > numerical_df.shape[1]
        ):

            n_components = (
                numerical_df.shape[1]
            )

        pca = PCA(
            n_components=n_components
        )

        transformed_data = (
            pca.fit_transform(
                numerical_df
            )
        )

        pca_columns = [
            f"PC{i+1}"
            for i in range(
                n_components
            )
        ]

        pca_df = pd.DataFrame(
            transformed_data,
            columns=pca_columns
        )

        # Force numeric dtype
        pca_df = (
            pca_df.astype(float)
        )

        explained_variance = (
            pca.explained_variance_ratio_
            .tolist()
        )

        return (
            pca_df,
            explained_variance
        )

    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2
    ):
        """
        Train-test split.
        """

        # Remove NaN values
        df = df.dropna()

        # Keep only numerical columns
        numerical_df = df.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )

        if target_column not in numerical_df.columns:

            raise ValueError(
                f"Target column '{target_column}' "
                "must be numerical."
            )

        x = numerical_df.drop(
            columns=[target_column]
        )

        y = numerical_df[target_column]

        x_train, x_test, y_train, y_test = (
            train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=42
            )
        )

        return {
            "x_train_shape":
                x_train.shape,

            "x_test_shape":
                x_test.shape,

            "y_train_shape":
                y_train.shape,

            "y_test_shape":
                y_test.shape
        }

    @staticmethod
    def feature_importance(
        df: pd.DataFrame,
        target_column: str
    ):
        """
        Generate feature importance.
        """

        x = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        # Remove NaN values
        combined_df = pd.concat(
            [x, y],
            axis=1
        ).dropna()

        x = combined_df.drop(
            columns=[target_column]
        )

        y = combined_df[target_column]

        # Keep only numerical features
        x = x.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )

        if x.empty:

            raise ValueError(
                "No numerical features available "
                "for feature importance."
            )

        # Detect problem type
        is_classification = False

        if (
            y.dtype == "object"
            or y.dtype.name == "category"
        ):
            is_classification = True

        elif (
            y.nunique() < 20
            and all(
                float(value).is_integer()
                for value in y.unique()
            )
        ):
            is_classification = True

        # Select model
        if is_classification:

            model = RandomForestClassifier(
                random_state=42
            )

        else:

            model = RandomForestRegressor(
                random_state=42
            )

        model.fit(x, y)

        importance_df = pd.DataFrame(
            {
                "Feature": x.columns,
                "Importance":
                    model.feature_importances_
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        return (
            importance_df
            .to_dict(
                orient="records"
            )
        )

    @staticmethod
    def save_transformed_dataset(
        df: pd.DataFrame
    ):
        """
        Save transformed dataset.
        """

        os.makedirs(
            TRANSFORMED_DATA_DIR,
            exist_ok=True
        )

        file_path = os.path.join(
            TRANSFORMED_DATA_DIR,
            "transformed_dataset.csv"
        )

        # Force scalar numeric conversion
        for column in df.columns:

            df[column] = (
                df[column]
                .apply(
                    lambda x:
                    x[0]
                    if isinstance(
                        x,
                        (list, np.ndarray)
                    )
                    else x
                )
            )

            df[column] = pd.to_numeric(
                df[column],
                errors="ignore"
            )

        df.to_csv(
            file_path,
            index=False
        )

        return file_path