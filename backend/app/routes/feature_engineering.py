"""
Feature engineering routes.
"""

import os

from fastapi import (
    APIRouter,
    HTTPException
)

from fastapi.responses import (
    FileResponse
)

from app.services.upload_service import (
    UploadService
)

from app.services.feature_engineering_service import (
    FeatureEngineeringService
)

router = APIRouter()

CLEANED_DATASET_PATH = (
    "app/reports/cleaned_dataset.csv"
)


@router.post("/feature-engineering")
async def feature_engineering(
    target_column: str,
    correlation_threshold: float = 0.9,
    variance_threshold: float = 0.01,
    pca_components: int = 2
):
    """
    Feature engineering pipeline.
    """

    try:

        # Load cleaned dataset instead
        if not os.path.exists(
            CLEANED_DATASET_PATH
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "Cleaned dataset not found. "
                    "Run data cleaning first."
                )
            )

        df = UploadService.load_dataset(
            CLEANED_DATASET_PATH
        )

        numerical_df = (
            df.select_dtypes(
                include=[
                    "int64",
                    "float64"
                ]
            )
        )

        if target_column not in numerical_df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Target column must "
                    "be numerical for "
                    "feature engineering."
                )
            )

        # Remove correlated features
        reduced_df, removed_corr = (
            FeatureEngineeringService
            .remove_correlated_features(
                numerical_df,
                correlation_threshold
            )
        )

        # Ensure target column exists
        if target_column not in reduced_df.columns:

            reduced_df[target_column] = (
                numerical_df[target_column]
            )

        # Separate target column
        target_data = reduced_df[
            target_column
        ]

        feature_df = reduced_df.drop(
            columns=[target_column]
        )

        # Remove low variance features
        variance_df, removed_variance = (
            FeatureEngineeringService
            .variance_threshold_selection(
                feature_df,
                variance_threshold
            )
        )

        # Add target column back
        variance_df[target_column] = (
            target_data.values
        )

        # Apply PCA only on features
        pca_df, explained_variance = (
            FeatureEngineeringService
            .apply_pca(
                variance_df.drop(
                    columns=[target_column]
                ),
                pca_components
            )
        )

        # Train-test split
        split_info = (
            FeatureEngineeringService
            .split_dataset(
                variance_df,
                target_column
            )
        )

        # Feature importance
        importance = (
            FeatureEngineeringService
            .feature_importance(
                variance_df,
                target_column
            )
        )

        # Save transformed dataset
        transformed_path = (
            FeatureEngineeringService
            .save_transformed_dataset(
                variance_df
            )
        )

        return {
            "status": "success",

            "removed_correlated_features":
                removed_corr,

            "removed_low_variance_features":
                removed_variance,

            "pca_explained_variance":
                explained_variance,

            "train_test_split":
                split_info,

            "feature_importance":
                importance,

            "transformed_dataset_path":
                transformed_path,

            "remaining_features":
                list(
                    variance_df.columns
                )
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get(
    "/download-transformed-data"
)
async def download_transformed_data():
    """
    Download transformed dataset.
    """

    file_path = (
        "app/reports/"
        "transformed_dataset.csv"
    )

    if not os.path.exists(file_path):

        raise HTTPException(
            status_code=404,
            detail=(
                "Transformed dataset "
                "not found."
            )
        )

    return FileResponse(
        path=file_path,
        filename="transformed_dataset.csv",
        media_type="text/csv"
    )