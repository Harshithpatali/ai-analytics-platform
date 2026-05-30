"""
Upload service for dataset handling.
"""

import os
import pandas as pd
from fastapi import UploadFile
from typing import Dict, Any

from app.utils.helper import (
    generate_timestamp_filename,
    ensure_directory_exists
)

UPLOAD_DIR = "app/uploads"

ensure_directory_exists(UPLOAD_DIR)


class UploadService:
    """
    Handles dataset uploads and summaries.
    """

    @staticmethod
    async def save_file(file: UploadFile) -> str:
        """
        Save uploaded file.
        """

        filename = generate_timestamp_filename(file.filename)

        file_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        contents = await file.read()

        with open(file_path, "wb") as uploaded_file:
            uploaded_file.write(contents)

        return file_path

    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """
        Load dataset from file.
        """

        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)

        if file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)

        raise ValueError("Unsupported file format.")

    @staticmethod
    def generate_dataset_summary(
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate dataset metadata summary.
        """

        summary = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "data_types": {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            },
            "missing_values": {
                col: int(value)
                for col, value in df.isnull().sum().items()
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 * 1024),
                2
            ),
            "numerical_columns": list(
                df.select_dtypes(
                    include=["int64", "float64"]
                ).columns
            ),
            "categorical_columns": list(
                df.select_dtypes(
                    include=["object", "category"]
                ).columns
            ),
        }

        return summary