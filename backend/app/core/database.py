"""
Database configuration.
"""

import os

from dotenv import load_dotenv

from sqlalchemy import create_engine

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

# Create engine
engine = create_engine(
    DATABASE_URL
)

# Session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base model
Base = declarative_base()


def get_db():
    """
    Database dependency.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()