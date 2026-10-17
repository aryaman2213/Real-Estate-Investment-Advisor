"""Step 2 of the project: train, compare, tune and register both models with MLflow.

Run:  python train.py            (after python pipeline.py)
View: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

import config
from src import modeling, tracking

CLF_TUNING_GRID = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5, 10],
    "model__reg_lambda": [0.5, 1, 2, 5],
}
# Same search space for the regressor, which sits inside TransformedTargetRegressor.
REG_TUNING_GRID = {k.replace("model__", "regressor__model__"): v for k, v in CLF_TUNING_GRID.items()}
REG_TUNING_GRID["regressor__model__n_estimators"] = [400, 600, 900]


def load_split():
    df = pd.read_parquet(config.APP_DATA_PATH)
    X = df[modeling.feature_columns()].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    y_clf, y_reg = df[config.CLASSIFICATION_TARGET], df[config.REGRESSION_TARGET]
    # One split, stratified on the class label, shared by both tasks.
    idx_train, idx_test = train_test_split(
        df.index, test_size=config.TEST_SIZE, stratify=y_clf, random_state=config.RANDOM_SEED)
    return df, X, y_clf, y_reg, idx_train, idx_test


def tune(pipeline, grid, X, y, scoring, stratify):
    """Randomised search on a sample (the full set is 200k rows), cross-validated."""
    sample = X.sample(n=min(config.TUNING_SAMPLE_SIZE, len(X)), random_state=config.RANDOM_SEED).index
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED) if stratify else config.CV_FOLDS
    search = RandomizedSearchCV(
        pipeline, grid, n_iter=config.TUNING_ITERATIONS, scoring=scoring, cv=cv,
        random_state=config.RANDOM_SEED, n_jobs=1, verbose=0)
    search.fit(X.loc[sample], y.loc[sample])
    return search


def run_classification(X, y, idx_train, idx_test) -> dict:
    tracking.setup_experiment(config.EXPERIMENT_CLASSIFICATION)
    X_train, X_test, y_train, y_test = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    results, fitted_models = {}, {}

    def evaluate_and_log(name, pipe, extra_params=None):
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = modeling.classification_metrics(y_test, pred, proba)
        metrics["train_seconds"] = train_seconds
        matrix = modeling.confusion(y_test, pred)
        importance = modeling.feature_importance(pipe)
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"task": "classification", "target": config.CLASSIFICATION_TARGET})
            mlflow.log_params({**tracking.loggable_params(pipe), **(extra_params or {}),
                               "train_rows": len(X_train), "test_rows": len(X_test)})
            mlflow.log_metrics(metrics)
            tracking.log_confusion_matrix(matrix, name)
            tracking.log_importance(importance, name)
            tracking.log_model(pipe, X_test.head(200))
        results[name] = {"run_id": run.info.run_id, "metrics": metrics, "confusion_matrix": matrix,
                         "feature_importance": None if importance is None else importance.round(5).to_dict()}
        fitted_models[name] = (pipe, proba, pred)
        print(f"  [clf] {name:<24} AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  ({train_seconds:.0f}s)")

    evaluate_and_log("Baseline (majority class)", modeling.classification_pipeline(modeling.baseline_classifier()))
    for name, model in modeling.classifiers(pos_weight).items():
        evaluate_and_log(name, modeling.classification_pipeline(model))

    search = tune(modeling.classification_pipeline(modeling.classifiers(pos_weight)["XGBoost"]),
                  CLF_TUNING_GRID, X_train, y_train, "roc_auc", stratify=True)
    tuned = modeling.classification_pipeline(modeling.classifiers(pos_weight)["XGBoost"])
    tuned.set_params(**search.best_params_)
    evaluate_and_log("XGBoost (tuned)", tuned,
                     {"cv_best_roc_auc": round(search.best_score_, 5), "tuning_iterations": config.TUNING_ITERATIONS})

    # Champion: highest ROC AUC (threshold-free, robust to the 73:27 imbalance); F1 breaks ties.
    candidates = {k: v for k, v in results.items() if not k.startswith("Baseline")}
    champion = max(candidates, key=lambda k: (round(candidates[k]["metrics"]["roc_auc"], 4),
                                              candidates[k]["metrics"]["f1"]))
    version = tracking.register_champion(
        results[champion]["run_id"], config.REGISTERED_CLASSIFIER,
        "Predicts whether a property is a Good Investment (beats 8% growth and is priced at or below its city median).")
    pipe, proba, pred = fitted_models[champion]
    joblib.dump(pipe, config.CLASSIFIER_PATH, compress=3)

    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
    return {
        "results": results, "champion": champion, "registry_version": version, "positive_rate": float(y.mean()),
        "pos_weight": pos_weight, "tuning_best_params": search.best_params_,
        "roc_curve": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
    }


def run_regression(df, X, y, idx_train, idx_test) -> dict:
    tracking.setup_experiment(config.EXPERIMENT_REGRESSION)
    X_train, X_test, y_train, y_test = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]
    results, predictions, fitted_models = {}, {}, {}

    # Baseline: the brief's "Fixed Rate" method, Future = Current x 1.08^5, no learning at all.
    naive = df.loc[idx_test, "Price_in_Lakhs"] * (1 + config.GOOD_INVESTMENT_MIN_GROWTH / 100) ** config.FORECAST_YEARS
    naive_metrics = modeling.regression_metrics(y_test, naive)
    with mlflow.start_run(run_name="Baseline (fixed 8% rate)") as run:
        mlflow.set_tags({"task": "regression", "target": config.REGRESSION_TARGET})
        mlflow.log_params({"model_class": "FixedRateFormula", "annual_rate_pct": config.GOOD_INVESTMENT_MIN_GROWTH})
        mlflow.log_metrics(naive_metrics)
        tracking.log_regression_plots(y_test, naive, "Fixed 8% baseline")
    results["Baseline (fixed 8% rate)"] = {"run_id": run.info.run_id, "metrics": naive_metrics, "feature_importance": None}
    print(f"  [reg] {'Baseline (fixed 8% rate)':<24} RMSE={naive_metrics['rmse']:.3f}  R2={naive_metrics['r2']:.5f}")

    def evaluate_and_log(name, est, extra_params=None):
        start = time.perf_counter()
        est.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        pred = est.predict(X_test)
        metrics = modeling.regression_metrics(y_test, pred)
        metrics["train_seconds"] = train_seconds
        importance = modeling.feature_importance(est)
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"task": "regression", "target": config.REGRESSION_TARGET, "target_transform": "log"})
            mlflow.log_params({**tracking.loggable_params(est), **(extra_params or {}),
                               "train_rows": len(X_train), "test_rows": len(X_test)})
            mlflow.log_metrics(metrics)
            tracking.log_regression_plots(y_test, pred, name)
            tracking.log_importance(importance, name)
            tracking.log_model(est, X_test.head(200))
        results[name] = {"run_id": run.info.run_id, "metrics": metrics,
                         "feature_importance": None if importance is None else importance.round(5).to_dict()}
        predictions[name] = pred
        fitted_models[name] = est
        print(f"  [reg] {name:<24} RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
              f"R2={metrics['r2']:.5f}  ({train_seconds:.0f}s)")

    for name, model in modeling.regressors().items():
        evaluate_and_log(name, modeling.regression_pipeline(model))

    search = tune(modeling.regression_pipeline(modeling.regressors()["XGBoost"]),
                  REG_TUNING_GRID, X_train, y_train, "neg_root_mean_squared_error", stratify=False)
    tuned = modeling.regression_pipeline(modeling.regressors()["XGBoost"])
    tuned.set_params(**search.best_params_)
    evaluate_and_log("XGBoost (tuned)", tuned,
                     {"cv_best_rmse": round(-search.best_score_, 5), "tuning_iterations": config.TUNING_ITERATIONS})

    candidates = {k: v for k, v in results.items() if not k.startswith("Baseline")}
    champion = min(candidates, key=lambda k: candidates[k]["metrics"]["rmse"])
    version = tracking.register_champion(
        results[champion]["run_id"], config.REGISTERED_REGRESSOR,
        "Predicts the property price after 5 years (lakhs). Trained on log(price) targets.")
    joblib.dump(fitted_models[champion], config.REGRESSOR_PATH, compress=3)

    # Prediction interval for the app: spread of log residuals on unseen data.
    pred = predictions[champion]
    log_resid = np.log(y_test.to_numpy()) - np.log(pred)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(pred), size=3000, replace=False)
    return {
        "results": results, "champion": champion, "registry_version": version,
        "tuning_best_params": {k.replace("regressor__", ""): v for k, v in search.best_params_.items()},
        "log_residual_quantiles": {"p05": float(np.quantile(log_resid, 0.05)), "p95": float(np.quantile(log_resid, 0.95))},
        "test_sample": {"actual": y_test.to_numpy()[sample].round(2).tolist(),
                        "predicted": pred[sample].round(2).tolist()},
    }


def main() -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    df, X, y_clf, y_reg, idx_train, idx_test = load_split()
    print(f"Train {len(idx_train):,} rows / test {len(idx_test):,} rows, {X.shape[1]} input features")
    clf = run_classification(X, y_clf, idx_train, idx_test)
    reg = run_regression(df, X, y_reg, idx_train, idx_test)

    summary = {
        "classification": {"champion": clf["champion"], "registry_version": clf["registry_version"],
                           "positive_rate": clf["positive_rate"], "pos_weight": clf["pos_weight"],
                           "tuning_best_params": clf["tuning_best_params"],
                           "models": {k: v["metrics"] for k, v in clf["results"].items()}},
        "regression": {"champion": reg["champion"], "registry_version": reg["registry_version"],
                       "tuning_best_params": reg["tuning_best_params"],
                       "models": {k: v["metrics"] for k, v in reg["results"].items()}},
        "train_rows": len(idx_train), "test_rows": len(idx_test),
    }
    details = {
        "classification": {"confusion_matrices": {k: v["confusion_matrix"] for k, v in clf["results"].items()},
                           "feature_importance": {k: v["feature_importance"] for k, v in clf["results"].items()},
                           "roc_curve": clf["roc_curve"]},
        "regression": {"feature_importance": {k: v["feature_importance"] for k, v in reg["results"].items()},
                       "log_residual_quantiles": reg["log_residual_quantiles"],
                       "test_sample": reg["test_sample"]},
    }
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2, default=float))
    config.EVALUATION_PATH.write_text(json.dumps(details, default=float))
    print(f"Classifier champion: {clf['champion']} (registry v{clf['registry_version']})")
    print(f"Regressor champion:  {reg['champion']} (registry v{reg['registry_version']})")


if __name__ == "__main__":
    main()
