"""
Data cleaning routes.
"""

import os

from fastapi import (
    APIRouter,
    HTTPException
)

from fastapi.responses import FileResponse

from app.services.upload_service import (
    UploadService
)

from app.services.preprocessing import (
    PreprocessingService
)

router = APIRouter()

UPLOAD_DIR = "app/uploads"


@router.post("/clean-data")
async def clean_data(
    missing_strategy: str = "mean",
    scaling_method: str = "standard"
):
    """
    Automated data cleaning endpoint.
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

        df = UploadService.load_dataset(
            file_path
        )

        cleaning_report = (
            PreprocessingService
            .clean_dataset(
                df,
                missing_strategy,
                scaling_method
            )
        )

        return {
            "status": "success",
            "cleaning_report": cleaning_report
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
    
@router.get("/download-cleaned-data")
async def download_cleaned_data():
    """
    Download cleaned dataset.
    """

    file_path = (
        "app/reports/cleaned_dataset.csv"
    )

    if not os.path.exists(file_path):

        raise HTTPException(
            status_code=404,
            detail="Cleaned dataset not found."
        )

    return FileResponse(
        path=file_path,
        filename="cleaned_dataset.csv",
        media_type="text/csv"
    )