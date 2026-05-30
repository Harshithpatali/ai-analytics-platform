"""
API client for backend communication.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv( "BACKEND_URL", "https://ai-analytics-platform-rgi6.onrender.com" )


class APIClient:
    """
    Handles backend API requests.
    """

    @staticmethod
    def get_health():
        """
        Check backend health.
        """

        try:

            response = requests.get(
                f"{BACKEND_URL}/api/health",
                timeout=10
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def upload_dataset(file):
        """
        Upload dataset to backend.
        """

        try:

            files = {
                "file": (
                    file.name,
                    file,
                    file.type
                )
            }

            response = requests.post(
                f"{BACKEND_URL}/api/upload",
                files=files,
                timeout=300
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def get_dataset_summary():
        """
        Fetch dataset summary.
        """

        try:

            response = requests.get(
                f"{BACKEND_URL}/api/dataset-summary",
                timeout=30
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def clean_dataset(
        missing_strategy,
        scaling_method
    ):
        """
        Trigger automated data cleaning.
        """

        try:

            response = requests.post(
                f"{BACKEND_URL}/api/clean-data",
                params={
                    "missing_strategy":
                        missing_strategy,

                    "scaling_method":
                        scaling_method
                },
                timeout=300
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def get_eda_summary():
        """
        Fetch EDA summary.
        """

        try:

            response = requests.get(
                f"{BACKEND_URL}/api/eda-summary",
                timeout=30
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def get_chart(
        endpoint,
        params=None
    ):
        """
        Fetch Plotly chart.
        """

        try:

            response = requests.get(
                f"{BACKEND_URL}/api/{endpoint}",
                params=params,
                timeout=30
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def feature_engineering(
        target_column,
        correlation_threshold,
        variance_threshold,
        pca_components
    ):
        """
        Run feature engineering.
        """

        try:

            response = requests.post(
                f"{BACKEND_URL}/api/feature-engineering",
                params={
                    "target_column":
                        target_column,

                    "correlation_threshold":
                        correlation_threshold,

                    "variance_threshold":
                        variance_threshold,

                    "pca_components":
                        pca_components
                },
                timeout=300
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def download_transformed_dataset():
        """
        Download transformed dataset.
        """

        try:

            response = requests.get(
                f"{BACKEND_URL}"
                "/api/download-transformed-data",
                timeout=300
            )

            return response

        except Exception as error:

            return {
                "status": "error",
                "message": str(error)
            }

    @staticmethod
    def train_models(
        target_column
    ):
        """
        Train machine learning models.
        """

        try:

            response = requests.post(
                f"{BACKEND_URL}/api/train",
                params={
                    "target_column":
                        target_column
                },
                timeout=1200
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "detail": str(error)
            }

    @staticmethod
    def compare_models(
        target_column
    ):
        """
        Compare machine learning models.
        """

        try:

            response = requests.post(
                f"{BACKEND_URL}/api/train",
                params={
                    "target_column":
                        target_column
                },
                timeout=1200
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "detail": str(error)
            }

    @staticmethod
    def predict(
        model_name
    ):
        """
        Generate predictions.
        """

        try:

            response = requests.post(
                f"{BACKEND_URL}/api/predict",
                params={
                    "model_name":
                        model_name
                },
                timeout=1200
            )

            return response.json()

        except Exception as error:

            return {
                "status": "error",
                "detail": str(error)
            }

    @staticmethod
    def download_predictions():
        """
        Download prediction CSV.
        """

        try:

            response = requests.get(
                f"{BACKEND_URL}"
                "/api/download-predictions"
            )

            return response

        except Exception as error:

            return {
                "status": "error",
                "detail": str(error)
            }