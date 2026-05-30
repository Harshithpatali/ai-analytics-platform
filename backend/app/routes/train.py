"""
Model training routes.
"""

import os

from fastapi import (
    APIRouter,
    HTTPException
)

from app.services.upload_service import (
    UploadService
)

from app.services.model_training import (
    ModelTrainingService
)

router = APIRouter()

CLEANED_DATASET_PATH = (
    "app/reports/cleaned_dataset.csv"
)

TRANSFORMED_DATASET_PATH = (
    "app/reports/transformed_dataset.csv"
)


@router.post("/train")
async def train_models(
    target_column: str
):
    """
    Train ML models.
    """

    try:

        # Priority:
        # 1. Transformed dataset
        # 2. Cleaned dataset

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

            raise HTTPException(
                status_code=404,
                detail=(
                    "Run data cleaning first."
                )
            )

        df = UploadService.load_dataset(
            dataset_path
        )

        if (
            target_column
            not in df.columns
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid target column."
                )
            )

        training_results = (
            ModelTrainingService
            .train_models(
                df,
                target_column
            )
        )

        return {
            "status": "success",

            "training_results":
                training_results
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )