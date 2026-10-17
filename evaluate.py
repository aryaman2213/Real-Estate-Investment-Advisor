"""Step 3 of the project: evaluate the registered champion models on the held-out test set.

Run:  python evaluate.py         (after python train.py)

Loads both models from the MLflow Model Registry by alias (models:/<name>@champion), recreates the same
train/validation/test split, and prints the full evaluation: classification report and confusion matrix for the
classifier, RMSE/MAE/MSE/R2 for the regressor. It also checks that the exported models/*.joblib files used by the
Streamlit app make exactly the same predictions as the registry versions.
"""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report

import config
from mlflow_ui import relocate_store
from src import modeling
from train import model_matrix, split_indices


def load_champion(name: str):
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, config.CHAMPION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{name}@{config.CHAMPION_ALIAS}")
    run = client.get_run(version.run_id)
    return model, version, run


def main() -> None:
    relocate_store()  # make artifact paths valid if the repo was cloned to a new folder
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    df = pd.read_parquet(config.APP_DATA_PATH)
    X = model_matrix(df)
    _, _, idx_test = split_indices(df)
    X_test = X.loc[idx_test]
    saved = json.loads(config.METRICS_PATH.read_text())

    clf, version, run = load_champion(config.REGISTERED_CLASSIFIER)
    threshold = float(run.data.params["decision_threshold"])
    y_test = df.loc[idx_test, config.CLASSIFICATION_TARGET]
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = modeling.classification_metrics(y_test, pred, proba)
    print(f"=== {config.REGISTERED_CLASSIFIER} v{version.version} (@{config.CHAMPION_ALIAS}): {run.info.run_name}")
    print(f"Decision threshold {threshold:.4f} | test rows {len(X_test):,}")
    print(classification_report(y_test, pred, target_names=["Not good", "Good investment"], digits=4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:", modeling.confusion(y_test, pred))
    print({k: round(v, 4) for k, v in metrics.items()})
    expected = saved["classification"]["models"][run.info.run_name]
    assert all(np.isclose(metrics[k], expected[k]) for k in metrics), "classifier metrics differ from train.py"
    assert np.allclose(joblib.load(config.CLASSIFIER_PATH).predict_proba(X_test)[:, 1], proba), "app model differs"

    reg, version, run = load_champion(config.REGISTERED_REGRESSOR)
    y_test = df.loc[idx_test, config.REGRESSION_TARGET]
    pred = reg.predict(X_test)
    metrics = modeling.regression_metrics(y_test, pred)
    print(f"\n=== {config.REGISTERED_REGRESSOR} v{version.version} (@{config.CHAMPION_ALIAS}): {run.info.run_name}")
    print({k: round(v, 4) for k, v in metrics.items()})
    expected = saved["regression"]["models"][run.info.run_name]
    assert all(np.isclose(metrics[k], expected[k]) for k in metrics), "regressor metrics differ from train.py"
    assert np.allclose(joblib.load(config.REGRESSOR_PATH).predict(X_test), pred), "app model differs"
    print("\nOK: registry champions reproduce the reported test metrics, and the app's exported models match them.")


if __name__ == "__main__":
    main()
