"""Model zoo, preprocessing pipeline and evaluation metrics for both tasks."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import config

SEED = config.RANDOM_SEED


def feature_columns() -> list[str]:
    return config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Impute (a safety net for app input; the training data has no gaps), scale numbers,
    one-hot encode categories. Unseen categories at prediction time are ignored, not errors."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC_FEATURES), ("cat", categorical, config.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def classifiers(pos_weight: float) -> dict:
    """At least five classifiers (the brief's minimum), from simplest to strongest.

    The classes are roughly 73:27, so every model that supports it weights the minority class.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1, random_state=SEED),
    }


def regressors() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=-1, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.06, max_depth=6, subsample=0.9, colsample_bytree=0.9,
            n_jobs=-1, random_state=SEED),
    }


def baseline_classifier() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def classification_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def regression_pipeline(model) -> TransformedTargetRegressor:
    """Future price = price x (1 + r)^5 is multiplicative; in log space it becomes additive
    (log F = log P + 5 log(1 + r)), which every model family can fit far more easily."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", build_preprocessor()), ("model", model)]),
        func=np.log, inverse_func=np.exp,
    )


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).tolist()


def feature_importance(fitted) -> pd.Series | None:
    """Importance per input column, one-hot columns summed back to their source feature.
    Tree models: impurity importance. Linear models: |coefficient| on scaled inputs."""
    pipe = fitted.regressor_ if isinstance(fitted, TransformedTargetRegressor) else fitted
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return None
    series = pd.Series(values, index=names)
    source = {}
    for name in names:
        if name in config.NUMERIC_FEATURES:  # e.g. City_Tier must not fold into City
            source[name] = name
        else:
            source[name] = next((c for c in config.CATEGORICAL_FEATURES if name.startswith(c + "_")), name)
    grouped = series.groupby(source).sum()
    return (grouped / grouped.sum()).sort_values(ascending=False)
