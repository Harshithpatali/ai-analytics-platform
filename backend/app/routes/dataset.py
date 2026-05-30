"""
Dataset routes.
"""

import os
import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException
)

from app.services.upload_service import UploadService

router = APIRouter()

UPLOAD_DIR = "app/uploads"


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "success",
        "message": "Backend is running successfully."
    }


@router.get("/dataset-summary")
async def dataset_summary():
    """
    Return latest uploaded dataset summary.
    """

    try:

        files = os.listdir(UPLOAD_DIR)

        if not files:
            raise HTTPException(
                status_code=404,
                detail="No uploaded dataset found."
            )

        latest_file = sorted(files)[-1]

        file_path = os.path.join(
            UPLOAD_DIR,
            latest_file
        )

        df = UploadService.load_dataset(file_path)

        summary = (
            UploadService
            .generate_dataset_summary(df)
        )

        return {
            "status": "success",
            "dataset_name": latest_file,
            "summary": summary
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )