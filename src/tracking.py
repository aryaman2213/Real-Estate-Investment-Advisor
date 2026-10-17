"""MLflow helpers: experiment setup, per-run logging with diagnostic plots, model registry."""
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import config


def setup_experiment(name: str) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        config.MLFLOW_ARTIFACT_DIR.mkdir(exist_ok=True)
        mlflow.create_experiment(name, artifact_location=(config.MLFLOW_ARTIFACT_DIR / name).as_uri())
    mlflow.set_experiment(name)


def loggable_params(estimator) -> dict:
    """The final model's hyperparameters, keeping only simple values MLflow can store."""
    model = estimator
    if hasattr(model, "regressor"):
        model = model.regressor
    if hasattr(model, "named_steps"):
        model = model.named_steps["model"]
    params = {k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None}
    params["model_class"] = type(model).__name__
    return params


def _log_figure(fig, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        fig.savefig(path, dpi=110, bbox_inches="tight")
        mlflow.log_artifact(str(path), artifact_path="plots")
    plt.close(fig)


def log_confusion_matrix(matrix, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.4))
    sns.heatmap(np.array(matrix), annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    _log_figure(fig, "confusion_matrix.png")


def log_importance(importance: pd.Series | None, title: str) -> None:
    if importance is None:
        return
    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index, top.values, color="#4f46e5")
    ax.set(title=title, xlabel="Share of importance")
    _log_figure(fig, "feature_importance.png")
    mlflow.log_dict(importance.round(5).to_dict(), "feature_importance.json")


def log_regression_plots(y_true, y_pred, title: str) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(4000, len(y_true)), replace=False)
    yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(yt, yp, s=4, alpha=0.4, color="#4f46e5")
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    axes[0].plot(lim, lim, color="#e11d48", lw=1)
    axes[0].set(xlabel="Actual (lakhs)", ylabel="Predicted (lakhs)", title=f"{title}: actual vs predicted")
    axes[1].hist(yt - yp, bins=60, color="#0ea5e9")
    axes[1].set(xlabel="Residual (lakhs)", title="Residual distribution")
    _log_figure(fig, "actual_vs_predicted.png")


def log_model(fitted, X_sample: pd.DataFrame) -> None:
    signature = infer_signature(X_sample, fitted.predict(X_sample))
    mlflow.sklearn.log_model(fitted, name="model", signature=signature, input_example=X_sample.head(3),
                             serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)


def register_champion(run_id: str, registered_name: str, description: str) -> int:
    """Register the run's model and point the 'champion' alias at the new version."""
    version = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
    client = MlflowClient()
    client.update_registered_model(registered_name, description=description)
    client.set_registered_model_alias(registered_name, config.CHAMPION_ALIAS, version.version)
    return int(version.version)
