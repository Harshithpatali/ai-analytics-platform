"""
Frontend helper utilities.
"""

import streamlit as st


def set_page_config(title: str) -> None:
    """
    Configure Streamlit page settings.
    """

    st.set_page_config(
        page_title=title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def show_success(message: str) -> None:
    """
    Display success message.
    """

    st.success(message)


def show_error(message: str) -> None:
    """
    Display error message.
    """

    st.error(message)