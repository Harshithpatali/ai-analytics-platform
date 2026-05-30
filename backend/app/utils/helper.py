"""
Helper utility functions.
"""

import os
from datetime import datetime


def generate_timestamp_filename(filename: str) -> str:
    """
    Generate unique filename using timestamp.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{timestamp}_{filename}"


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it does not exist.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)