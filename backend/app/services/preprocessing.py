"""
Data preprocessing and cleaning service.
"""

import os
import pandas as pd
import numpy as np

from typing import Dict, Any

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

from app.utils.helper import (
    ensure_directory_exists
)

CLEANED_DATA_DIR = "app/reports"

ensure_directory_exists(
    CLEANED_DATA_DIR
)


class PreprocessingService:
    """
    Handles automated data cleaning.
    """

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values.
        """

        numerical_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_columns = (
            df.select_dtypes(
                exclude=np.number
            ).columns
        )

        for column in numerical_columns:

            if df[column].isnull().sum() > 0:

                if strategy == "mean":
                    value = df[column].mean()

                else:
                    value = df[column].median()

                df[column] = (
                    df[column]
                    .fillna(value)
                )

        for column in categorical_columns:

            if df[column].isnull().sum() > 0:

                mode_value = (
                    df[column]
                    .mode()[0]
                )

                df[column] = (
                    df[column]
                    .fillna(mode_value)
                )

        return df

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.
        """

        return df.drop_duplicates()

    @staticmethod
    def detect_outliers_iqr(
        df: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Detect outliers using IQR.
        """

        outlier_summary = {}

        numerical_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        for column in numerical_columns:

            q1 = df[column].quantile(0.25)

            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )

            outliers = df[
                (
                    df[column]
                    < lower_bound
                )
                |
                (
                    df[column]
                    > upper_bound
                )
            ]

            outlier_summary[column] = (
                outliers.shape[0]
            )

        return outlier_summary

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Cap outliers using IQR.
        """

        numerical_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        for column in numerical_columns:

            q1 = df[column].quantile(0.25)

            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )

            df[column] = np.where(
                df[column] < lower_bound,
                lower_bound,
                df[column]
            )

            df[column] = np.where(
                df[column] > upper_bound,
                upper_bound,
                df[column]
            )

        return df

    @staticmethod
    def encode_categorical_features(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode categorical columns.
        """

        categorical_columns = (
            df.select_dtypes(
                include=["object"]
            ).columns
        )

        encoder = LabelEncoder()

        for column in categorical_columns:

            df[column] = encoder.fit_transform(
                df[column].astype(str)
            )

        return df

    @staticmethod
    def scale_features(
        df: pd.DataFrame,
        scaling_method: str = "standard"
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        """

        numerical_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        if scaling_method == "standard":

            scaler = StandardScaler()

        else:

            scaler = MinMaxScaler()

        df[numerical_columns] = scaler.fit_transform(
            df[numerical_columns]
        )

        return df

    @staticmethod
    def save_cleaned_dataset(
        df: pd.DataFrame
    ) -> str:
        """
        Save cleaned dataset.
        """

        file_path = os.path.join(
            CLEANED_DATA_DIR,
            "cleaned_dataset.csv"
        )

        df.to_csv(
            file_path,
            index=False
        )

        return file_path

    @staticmethod
    def clean_dataset(
        df: pd.DataFrame,
        missing_strategy: str,
        scaling_method: str
    ) -> Dict[str, Any]:
        """
        Complete cleaning pipeline.
        """

        before_shape = df.shape

        before_missing = (
            int(df.isnull().sum().sum())
        )

        before_duplicates = (
            int(df.duplicated().sum())
        )

        outliers_before = (
            PreprocessingService
            .detect_outliers_iqr(df)
        )

        df = (
            PreprocessingService
            .handle_missing_values(
                df,
                missing_strategy
            )
        )

        df = (
            PreprocessingService
            .remove_duplicates(df)
        )

        df = (
            PreprocessingService
            .handle_outliers(df)
        )

        df = (
            PreprocessingService
            .encode_categorical_features(df)
        )

        df = (
            PreprocessingService
            .scale_features(
                df,
                scaling_method
            )
        )

        cleaned_file_path = (
            PreprocessingService
            .save_cleaned_dataset(df)
        )

        after_shape = df.shape

        after_missing = (
            int(df.isnull().sum().sum())
        )

        after_duplicates = (
            int(df.duplicated().sum())
        )

        outliers_after = (
            PreprocessingService
            .detect_outliers_iqr(df)
        )

        return {
            "before_shape": before_shape,
            "after_shape": after_shape,
            "before_missing": before_missing,
            "after_missing": after_missing,
            "before_duplicates": before_duplicates,
            "after_duplicates": after_duplicates,
            "outliers_before": outliers_before,
            "outliers_after": outliers_after,
            "cleaned_dataset_path": cleaned_file_path
        }