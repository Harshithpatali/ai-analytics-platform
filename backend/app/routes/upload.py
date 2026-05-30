"""
Dataset upload routes.
"""

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.services.upload_service import UploadService
from app.utils.validators import validate_file
from app.utils.logger import setup_logger

router = APIRouter()

logger = setup_logger(__name__)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):
    """
    Upload dataset endpoint.
    """

    try:

        logger.info(
            "Dataset upload started: %s",
            file.filename
        )

        validate_file(file)

        file_path = await UploadService.save_file(file)

        df = UploadService.load_dataset(file_path)

        summary = UploadService.generate_dataset_summary(df)

        logger.info(
            "Dataset uploaded successfully."
        )

        return {
            "status": "success",
            "filename": file.filename,
            "saved_path": file_path,
            "summary": summary
        }

    except Exception as error:

        logger.error(
            "Dataset upload failed: %s",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )