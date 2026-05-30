"""
Predictions dashboard.
"""

from io import BytesIO

import pandas as pd
import streamlit as st

from utils.helpers import (
    set_page_config
)

from utils.api_client import (
    APIClient
)

set_page_config(
    "Predictions"
)

st.title(
    "🔮 Prediction Engine"
)

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "SVM",
    "Linear Regression",
    "Gradient Boosting"
]

model_name = st.selectbox(
    "Select Model",
    models
)

st.divider()

st.subheader(
    "🚀 Batch Predictions"
)

st.markdown(
    """
Generate predictions using the
selected trained model.
"""
)

if st.button(
    "Generate Predictions"
):

    with st.spinner(
        "Generating predictions..."
    ):

        response = (
            APIClient.predict(
                model_name
            )
        )

    if (
        response.get("status")
        == "success"
    ):

        st.success(
            "Predictions generated."
        )

        st.divider()

        st.subheader(
            "⬇ Download Predictions"
        )

        try:

            download_response = (
                APIClient
                .download_predictions()
            )

            if (
                download_response
                and
                download_response.status_code
                == 200
            ):

                try:

                    prediction_df = (
                        pd.read_csv(
                            BytesIO(
                                download_response.content
                            )
                        )
                    )

                    st.dataframe(
                        prediction_df.head(20),
                        use_container_width=True
                    )

                except Exception as error:

                    st.warning(
                        f"Failed to preview predictions: {error}"
                    )

                st.download_button(
                    label=(
                        "Download Predictions CSV"
                    ),
                    data=(
                        download_response.content
                    ),
                    file_name=(
                        "predictions.csv"
                    ),
                    mime="text/csv"
                )

            else:

                st.error(
                    "Failed to download predictions."
                )

        except Exception as error:

            st.error(
                f"Download error: {error}"
            )

    else:

        st.error(
            response.get(
                "detail",
                response.get(
                    "message",
                    "Prediction failed."
                )
            )
        )

        st.json(response)