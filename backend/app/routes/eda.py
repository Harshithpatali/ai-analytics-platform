"""
EDA routes.
"""

import os

from fastapi import (
    APIRouter,
    HTTPException
)

from app.services.upload_service import (
    UploadService
)

from app.services.eda_service import (
    EDAService
)

from app.services.visualization_service import (
    VisualizationService
)

router = APIRouter()

UPLOAD_DIR = "app/uploads"


@router.get("/eda-summary")
async def eda_summary():
    """
    Return EDA summary.
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

        summary = (
            EDAService
            .generate_eda_summary(df)
        )

        return {
            "status": "success",
            "eda_summary": summary
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/histogram")
async def histogram(
    column: str
):
    """
    Histogram visualization.
    """

    files = os.listdir(UPLOAD_DIR)

    latest_file = sorted(files)[-1]

    file_path = os.path.join(
        UPLOAD_DIR,
        latest_file
    )

    df = UploadService.load_dataset(
        file_path
    )

    chart = (
        VisualizationService
        .histogram(df, column)
    )

    return {
        "chart": chart
    }


@router.get("/boxplot")
async def boxplot(
    column: str
):
    """
    Boxplot visualization.
    """

    files = os.listdir(UPLOAD_DIR)

    latest_file = sorted(files)[-1]

    file_path = os.path.join(
        UPLOAD_DIR,
        latest_file
    )

    df = UploadService.load_dataset(
        file_path
    )

    chart = (
        VisualizationService
        .boxplot(df, column)
    )

    return {
        "chart": chart
    }


@router.get("/scatter-plot")
async def scatter_plot(
    x_column: str,
    y_column: str
):
    """
    Scatter plot visualization.
    """

    files = os.listdir(UPLOAD_DIR)

    latest_file = sorted(files)[-1]

    file_path = os.path.join(
        UPLOAD_DIR,
        latest_file
    )

    df = UploadService.load_dataset(
        file_path
    )

    chart = (
        VisualizationService
        .scatter_plot(
            df,
            x_column,
            y_column
        )
    )

    return {
        "chart": chart
    }


@router.get("/correlation-heatmap")
async def correlation_heatmap():
    """
    Correlation heatmap visualization.
    """

    files = os.listdir(UPLOAD_DIR)

    latest_file = sorted(files)[-1]

    file_path = os.path.join(
        UPLOAD_DIR,
        latest_file
    )

    df = UploadService.load_dataset(
        file_path
    )

    chart = (
        VisualizationService
        .correlation_heatmap(df)
    )

    return {
        "chart": chart
    }


@router.get("/missing-value-heatmap")
async def missing_value_heatmap():
    """
    Missing value heatmap visualization.
    """

    files = os.listdir(UPLOAD_DIR)

    latest_file = sorted(files)[-1]

    file_path = os.path.join(
        UPLOAD_DIR,
        latest_file
    )

    df = UploadService.load_dataset(
        file_path
    )

    chart = (
        VisualizationService
        .missing_value_heatmap(df)
    )

    return {
        "chart": chart
    }


@router.get("/categorical-distribution")
async def categorical_distribution(
    column: str
):
    """
    Categorical analysis visualization.
    """

    files = os.listdir(UPLOAD_DIR)

    latest_file = sorted(files)[-1]

    file_path = os.path.join(
        UPLOAD_DIR,
        latest_file
    )

    df = UploadService.load_dataset(
        file_path
    )

    chart = (
        VisualizationService
        .categorical_distribution(
            df,
            column
        )
    )

    return {
        "chart": chart
    }