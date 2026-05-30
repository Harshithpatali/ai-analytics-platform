"""
Main FastAPI application.
"""
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import (
    dataset,
    upload,
    cleaning,
    eda,
    feature_engineering,
    train,
    prediction
)

from app.utils.logger import setup_logger


logger = setup_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
# Include Routes
app.include_router(
    dataset.router,
    prefix=settings.API_PREFIX,
    tags=["Dataset"]
)

app.include_router(
    upload.router,
    prefix=settings.API_PREFIX,
    tags=["Upload"]
)

app.include_router(
    cleaning.router,
    prefix=settings.API_PREFIX,
    tags=["Cleaning"]
)
app.include_router(
    eda.router,
    prefix=settings.API_PREFIX,
    tags=["EDA"]
)
app.include_router(
    feature_engineering.router,
    prefix=settings.API_PREFIX,
    tags=["Feature Engineering"]
)
app.include_router(
    train.router,
    prefix=settings.API_PREFIX,
    tags=["Training"]
)
app.include_router(
    prediction.router,
    prefix=settings.API_PREFIX,
    tags=["Prediction"]
)
@app.get("/")
async def root():
    """
    Root endpoint.
    """

    logger.info("Root endpoint accessed.")

    return {
        "message": "AI Analytics Platform Backend Running"
    }