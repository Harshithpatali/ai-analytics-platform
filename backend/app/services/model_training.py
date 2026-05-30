"""
Machine learning training service.
"""

import json
import os
import joblib
import pandas as pd

from typing import Dict, Any

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from lightgbm import (
    LGBMClassifier
)

MODEL_DIR = "app/models"

METADATA_DIR = (
    "app/models/metadata"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    METADATA_DIR,
    exist_ok=True
)


class ModelTrainingService:
    """
    Handles machine learning training.
    """

    @staticmethod
    def detect_problem_type(
        y: pd.Series
    ) -> str:
        """
        Detect classification or regression.
        """

        if (
            y.dtype == "object"
            or y.dtype.name == "category"
        ):
            return "classification"

        if (
            y.nunique() < 20
            and all(
                float(value).is_integer()
                for value in y.unique()
            )
        ):
            return "classification"

        return "regression"

    @staticmethod
    def get_classification_models():
        """
        Classification models.
        """

        return {
            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=50,
                    n_jobs=-1,
                    random_state=42
                ),

            "XGBoost":
                XGBClassifier(
                    n_estimators=50,
                    random_state=42
                ),

            "LightGBM":
                LGBMClassifier(
                    n_estimators=50,
                    random_state=42
                ),

            "SVM":
                SVC(
                    probability=True
                )
        }

    @staticmethod
    def get_regression_models():
        """
        Regression models.
        """

        return {
            "Linear Regression":
                LinearRegression(),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=50,
                    n_jobs=-1,
                    random_state=42
                ),

            "XGBoost":
                XGBRegressor(
                    n_estimators=50,
                    random_state=42
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                )
        }

    @staticmethod
    def classification_metrics(
        y_true,
        y_pred,
        y_proba=None
    ):
        """
        Classification metrics.
        """

        metrics = {
            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

            "f1_score":
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

            "confusion_matrix":
                confusion_matrix(
                    y_true,
                    y_pred
                ).tolist()
        }

        if y_proba is not None:

            try:

                metrics["roc_auc"] = (
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr"
                    )
                )

            except Exception:

                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def regression_metrics(
        y_true,
        y_pred
    ):
        """
        Regression metrics.
        """

        mse = mean_squared_error(
            y_true,
            y_pred
        )

        return {
            "mae":
                mean_absolute_error(
                    y_true,
                    y_pred
                ),

            "mse": mse,

            "rmse":
                mse ** 0.5,

            "r2_score":
                r2_score(
                    y_true,
                    y_pred
                )
        }

    @staticmethod
    def generate_classification_charts(
        y_true,
        y_pred,
        y_proba=None
    ):
        """
        Generate classification analytics.
        """

        analytics = {}

        analytics["confusion_matrix"] = (
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
        )

        if y_proba is not None:

            try:

                fpr, tpr, _ = roc_curve(
                    y_true,
                    y_proba
                )

                analytics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }

            except Exception:

                analytics["roc_curve"] = None

        if y_proba is not None:

            try:

                precision, recall, _ = (
                    precision_recall_curve(
                        y_true,
                        y_proba
                    )
                )

                analytics[
                    "precision_recall_curve"
                ] = {
                    "precision":
                        precision.tolist(),

                    "recall":
                        recall.tolist()
                }

            except Exception:

                analytics[
                    "precision_recall_curve"
                ] = None

        return analytics

    @staticmethod
    def generate_regression_charts(
        y_true,
        y_pred
    ):
        """
        Generate regression analytics.
        """

        residuals = y_true - y_pred

        return {
            "actual_values":
                y_true.tolist(),

            "predicted_values":
                y_pred.tolist(),

            "residuals":
                residuals.tolist()
        }

    @staticmethod
    def recommend_best_model(
        results,
        problem_type
    ):
        """
        Recommend best model.
        """

        best_model = None

        best_score = -999999

        for model_name, result in (
            results.items()
        ):

            metrics = result["metrics"]

            if (
                problem_type
                == "classification"
            ):

                score = (
                    metrics["accuracy"]
                )

            else:

                score = (
                    metrics["r2_score"]
                )

            if score > best_score:

                best_score = score
                best_model = model_name

        return {
            "recommended_model":
                best_model,

            "score":
                best_score
        }

    @staticmethod
    def save_model(
        model,
        model_name: str
    ) -> str:
        """
        Save trained model.
        """

        file_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.pkl"
        )

        joblib.dump(
            model,
            file_path
        )

        return file_path

    @staticmethod
    def save_model_metadata(
        model_name,
        feature_columns,
        target_column
    ):
        """
        Save model metadata.
        """

        metadata = {
            "feature_columns":
                feature_columns,

            "target_column":
                target_column
        }

        metadata_path = os.path.join(
            METADATA_DIR,
            f"{model_name}.json"
        )

        with open(
            metadata_path,
            "w"
        ) as file:

            json.dump(
                metadata,
                file
            )

    @staticmethod
    def train_models(
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Train machine learning models.
        """

        df = df.dropna()

        x = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        # Keep numerical features only
        x = x.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )

        problem_type = (
            ModelTrainingService
            .detect_problem_type(y)
        )

        x_train, x_test, y_train, y_test = (
            train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        if (
            problem_type
            == "classification"
        ):

            models = (
                ModelTrainingService
                .get_classification_models()
            )

        else:

            models = (
                ModelTrainingService
                .get_regression_models()
            )

        results = {}

        best_score = -999999
        best_model_name = None

        for model_name, model in models.items():

            model.fit(
                x_train,
                y_train
            )

            predictions = model.predict(
                x_test
            )

            probabilities = None

            if (
                problem_type
                == "classification"
            ):

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            x_test
                        )
                    )

                    if (
                        len(
                            probabilities.shape
                        ) > 1
                        and probabilities.shape[1]
                        == 2
                    ):

                        probabilities = (
                            probabilities[:, 1]
                        )

                metrics = (
                    ModelTrainingService
                    .classification_metrics(
                        y_test,
                        predictions,
                        probabilities
                    )
                )

                score = (
                    metrics["accuracy"]
                )

                analytics = (
                    ModelTrainingService
                    .generate_classification_charts(
                        y_test,
                        predictions,
                        probabilities
                    )
                )

            else:

                metrics = (
                    ModelTrainingService
                    .regression_metrics(
                        y_test,
                        predictions
                    )
                )

                score = (
                    metrics["r2_score"]
                )

                analytics = (
                    ModelTrainingService
                    .generate_regression_charts(
                        y_test,
                        predictions
                    )
                )

            model_path = (
                ModelTrainingService
                .save_model(
                    model,
                    model_name
                )
            )

            ModelTrainingService.save_model_metadata(
                model_name,
                list(x.columns),
                target_column
            )

            results[model_name] = {
                "metrics": metrics,
                "analytics": analytics,
                "model_path": model_path
            }

            if score > best_score:

                best_score = score
                best_model_name = model_name

        recommendation = (
            ModelTrainingService
            .recommend_best_model(
                results,
                problem_type
            )
        )

        return {
            "problem_type":
                problem_type,

            "results":
                results,

            "best_model":
                best_model_name,

            "recommendation":
                recommendation
        }