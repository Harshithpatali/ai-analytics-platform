"""
Validation utilities.
"""

from fastapi import (
    UploadFile,
    HTTPException
)

ALLOWED_EXTENSIONS = [
    ".csv",
    ".xlsx"
]

MAX_FILE_SIZE_MB = 100


def validate_file(
    file: UploadFile
) -> None:
    """
    Validate uploaded dataset file.
    """

    filename = file.filename.lower()

    if not any(
        filename.endswith(ext)
        for ext in ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV and Excel "
                "files are allowed."
            )
        )

    if file.size:

        file_size_mb = (
            file.size / (1024 * 1024)
        )

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File size exceeds "
                    f"{MAX_FILE_SIZE_MB} MB."
                )
            )