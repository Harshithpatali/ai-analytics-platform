"""
Prediction service.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from app.services.upload_service import (
    UploadService
)

MODEL_DIR = (
    "app/models"
)

METADATA_DIR = (
    "app/models/metadata"
)

PREDICTION_OUTPUT_PATH = (
    "app/reports/predictions.csv"
)

TRANSFORMED_DATASET_PATH = (
    "app/reports/transformed_dataset.csv"
)

CLEANED_DATASET_PATH = (
    "app/reports/cleaned_dataset.csv"
)


class PredictionService:
    """
    Handles ML predictions.
    """

    @staticmethod
    def load_model(
        model_name: str
    ):
        """
        Load trained model.
        """

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.pkl"
        )

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                f"Model not found: "
                f"{model_name}"
            )

        return joblib.load(
            model_path
        )

    @staticmethod
    def load_metadata(
        model_name: str
    ):
        """
        Load model metadata.
        """

        metadata_path = os.path.join(
            METADATA_DIR,
            f"{model_name}.json"
        )

        if not os.path.exists(
            metadata_path
        ):

            raise FileNotFoundError(
                f"Metadata not found "
                f"for model: "
                f"{model_name}"
            )

        with open(
            metadata_path,
            "r"
        ) as file:

            return json.load(
                file
            )

    @staticmethod
    def load_prediction_dataset():
        """
        Load dataset for predictions.
        """

        if os.path.exists(
            TRANSFORMED_DATASET_PATH
        ):

            dataset_path = (
                TRANSFORMED_DATASET_PATH
            )

        elif os.path.exists(
            CLEANED_DATASET_PATH
        ):

            dataset_path = (
                CLEANED_DATASET_PATH
            )

        else:

            raise FileNotFoundError(
                "No cleaned/transformed "
                "dataset found."
            )

        return UploadService.load_dataset(
            dataset_path
        )

    @staticmethod
    def generate_predictions(
        model_name: str
    ):
        """
        Generate predictions.
        """

        model = (
            PredictionService
            .load_model(
                model_name
            )
        )

        metadata = (
            PredictionService
            .load_metadata(
                model_name
            )
        )

        feature_columns = (
            metadata[
                "feature_columns"
            ]
        )

        target_column = (
            metadata[
                "target_column"
            ]
        )

        df = (
            PredictionService
            .load_prediction_dataset()
        )

        prediction_df = (
            df.copy()
        )

        # Remove target column if exists
        if (
            target_column
            in prediction_df.columns
        ):

            prediction_df = (
                prediction_df.drop(
                    columns=[
                        target_column
                    ]
                )
            )

        # Keep required features only
        missing_features = []

        for feature in feature_columns:

            if (
                feature
                not in prediction_df.columns
            ):

                missing_features.append(
                    feature
                )

                prediction_df[
                    feature
                ] = 0

        prediction_df = (
            prediction_df[
                feature_columns
            ]
        )

        # Clean all values safely
        for column in prediction_df.columns:

            cleaned_values = []

            for value in prediction_df[column]:

                # Handle numpy arrays/lists
                if isinstance(
                    value,
                    (list, np.ndarray)
                ):

                    if len(value) > 0:

                        value = value[0]

                    else:

                        value = 0

                # Handle NaN values
                if pd.isna(value):

                    value = 0

                # Convert to string
                value = str(value)

                # Remove brackets/spaces
                value = (
                    value
                    .replace("[", "")
                    .replace("]", "")
                    .strip()
                )

                try:

                    value = float(value)

                except Exception:

                    value = 0.0

                cleaned_values.append(
                    value
                )

            prediction_df[column] = (
                cleaned_values
            )

        # Final cleanup
        prediction_df = (
            prediction_df.fillna(0)
        )

        prediction_df = (
            prediction_df.astype(
                "float64"
            )
        )

        print(
            "\nPrediction dataframe dtypes:\n",
            prediction_df.dtypes
        )

        print(
            "\nPrediction dataframe sample:\n",
            prediction_df.head()
        )

        # Generate predictions
        predictions = (
            model.predict(
                prediction_df
            )
        )

        prediction_output = (
            df.copy()
        )

        prediction_output[
            "Prediction"
        ] = predictions

        prediction_output.to_csv(
            PREDICTION_OUTPUT_PATH,
            index=False
        )

        return {
            "status": "success",

            "model_name":
                model_name,

            "prediction_count":
                len(predictions),

            "missing_features":
                missing_features,

            "prediction_file":
                PREDICTION_OUTPUT_PATH,

            "sample_predictions":
                prediction_output[
                    "Prediction"
                ]
                .head(10)
                .tolist()
        }