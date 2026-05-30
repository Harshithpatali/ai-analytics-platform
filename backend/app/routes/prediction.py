"""
Prediction routes.
"""

import os
import pandas as pd

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

from app.services.prediction_service import (
    PredictionService
)

from app.services.explainability_service import (
    ExplainabilityService
)

from app.services.ai_insights_service import (
    AIInsightsService
)

router = APIRouter()

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "app",
    "models"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "app",
    "uploads"
)

PREDICTION_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "app",
    "reports",
    "predictions.csv"
)


@router.post("/predict")
async def predict(
    model_name: str
):
    """
    Generate predictions.
    """

    try:

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.pkl"
        )

        print(
            "MODEL PATH:",
            model_path
        )

        if not os.path.exists(
            model_path
        ):

            raise HTTPException(
                status_code=404,
                detail="Model not found."
            )

        files = os.listdir(
            UPLOAD_DIR
        )

        if not files:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No uploaded dataset found."
                )
            )

        latest_file = sorted(files)[-1]

        dataset_path = os.path.join(
            UPLOAD_DIR,
            latest_file
        )

        df = UploadService.load_dataset(
            dataset_path
        )

        model = (
            PredictionService
            .load_model(
                model_name
            )
        )

        prediction_results = (
            PredictionService
            .generate_predictions(
                model_name
            )
        )

        transformed_df = (
            df.select_dtypes(
                include=[
                    "int64",
                    "float64",
                    "object"
                ]
            )
            .fillna(0)
        )

        # Clean numeric conversion safely
        clean_df = transformed_df.copy()

        for column in clean_df.columns:

            clean_df[column] = (
                clean_df[column]
                .astype(str)
                .str.replace(
                    "[",
                    "",
                    regex=False
                )
                .str.replace(
                    "]",
                    "",
                    regex=False
                )
                .str.strip()
            )

            clean_df[column] = (
                pd.to_numeric(
                    clean_df[column],
                    errors="coerce"
                )
            )

        clean_df = (
            clean_df.fillna(0)
        )

        clean_df = (
            clean_df.astype(
                "float64"
            )
        )

        # Safe feature importance
        explainability = (
            ExplainabilityService
            .generate_feature_importance(
                model,
                list(
                    transformed_df.columns
                )
            )
        )

        # AI Insights
        insights = (
            AIInsightsService
            .generate_insights(
                dataset_summary=str(
                    df.describe()
                ),
                model_metrics=str(
                    explainability[
                        "feature_importance"
                    ][:5]
                )
            )
        )

        return {
            "status": "success",

            "prediction_results":
                prediction_results,

            "explainability":
                explainability,

            "ai_insights":
                insights
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get(
    "/download-predictions"
)
async def download_predictions():
    """
    Download prediction CSV.
    """

    if not os.path.exists(
        PREDICTION_OUTPUT_PATH
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Prediction file not found."
            )
        )

    return FileResponse(
        path=PREDICTION_OUTPUT_PATH,
        filename="predictions.csv",
        media_type="text/csv"
    )