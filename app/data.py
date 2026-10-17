"""Data and model access for the app: cached loading, filtering, and predictions with explanations."""
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.features import engineer, load_market_reference
from src.modeling import feature_columns

APP_COLUMNS = [
    "ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    "Year_Built", "Age_of_Property", "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools",
    "Nearby_Hospitals", "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities", "Facing",
    "Owner_Type", "Availability_Status", "City_Tier", "PPSF_vs_City_Median", "Infrastructure_Score", "Amenity_Count",
    "Annual_Growth_Rate", "Future_Price_5Y", "Good_Investment", "Price_Category", "Age_Category",
]


ARTIFACTS = [config.APP_DATA_PATH, config.CLASSIFIER_PATH, config.REGRESSOR_PATH, config.METRICS_PATH,
             config.EVALUATION_PATH, config.MARKET_REFERENCE_PATH]


def _stamp() -> tuple:
    """Changes whenever a data or model file changes, so the caches below refresh after a redeploy
    (Streamlit Cloud pulls new files without restarting the server)."""
    return tuple(p.stat().st_mtime_ns for p in ARTIFACTS)


@st.cache_resource(show_spinner=False, max_entries=1)
def _models(stamp):
    return joblib.load(config.CLASSIFIER_PATH), joblib.load(config.REGRESSOR_PATH)


def models():
    return _models(_stamp())


@st.cache_data(show_spinner=False)
def _market_reference(stamp):
    return load_market_reference()


def market_reference():
    return _market_reference(_stamp())


@st.cache_data(show_spinner=False)
def _metrics(stamp):
    return json.loads(config.METRICS_PATH.read_text())


def metrics():
    return _metrics(_stamp())


def decision_threshold() -> float:
    """Probability cut-off for 'Good Investment', tuned on the validation split to maximise F1."""
    return float(metrics()["classification"]["threshold"])


@st.cache_data(show_spinner=False)
def _evaluation(stamp):
    return json.loads(config.EVALUATION_PATH.read_text())


def evaluation():
    return _evaluation(_stamp())


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_columns()].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    return X


@st.cache_data(show_spinner="Loading 250,000 listings and scoring them with the models...", max_entries=1)
def _listings(stamp) -> pd.DataFrame:
    df = pd.read_parquet(config.APP_DATA_PATH)
    clf, reg = models()
    X = model_input(df)
    out = df[APP_COLUMNS].copy()
    out["Model_Probability"] = clf.predict_proba(X)[:, 1]
    out["Model_Verdict"] = (out["Model_Probability"] >= decision_threshold()).astype(int)
    out["Predicted_Price_5Y"] = reg.predict(X)
    out["Predicted_Gain_Pct"] = (out["Predicted_Price_5Y"] / out["Price_in_Lakhs"] - 1) * 100
    out["Tier_Label"] = np.where(out["City_Tier"] == 1, "Tier 1 metro", "Tier 2 city")
    return out


def listings() -> pd.DataFrame:
    """The cleaned dataset, plus each listing's model scores (computed once per data/model version, then cached)."""
    return _listings(_stamp())


@dataclass(frozen=True)
class Filters:
    states: tuple = ()
    cities: tuple = ()
    property_types: tuple = ()
    bhk: tuple = ()
    price: tuple = (10, 500)
    size: tuple = (500, 5000)
    availability: tuple = ()
    good_only: bool = False

    def describe(self) -> str:
        parts = [", ".join(v) for v in (self.states, self.cities, self.property_types) if v]
        if self.bhk:
            parts.append("/".join(map(str, self.bhk)) + " BHK")
        if self.price != (10, 500):
            parts.append(f"Rs {self.price[0]}-{self.price[1]} L")
        if self.size != (500, 5000):
            parts.append(f"{self.size[0]:,}-{self.size[1]:,} sq ft")
        if self.availability:
            parts.append(", ".join(self.availability))
        if self.good_only:
            parts.append("Good Investments only")
        return " · ".join(parts) if parts else "All 250,000 listings"


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    mask = df["Price_in_Lakhs"].between(*f.price) & df["Size_in_SqFt"].between(*f.size)
    for col, values in [("State", f.states), ("City", f.cities), ("Property_Type", f.property_types),
                        ("BHK", f.bhk), ("Availability_Status", f.availability)]:
        if values:
            mask &= df[col].isin(values)
    if f.good_only:
        mask &= df["Good_Investment"] == 1
    return df.loc[mask]


# --- Single-property prediction (Advisor page) ----------------------------------------
def build_property(form: dict) -> pd.DataFrame:
    """Turn the form's raw inputs into a one-row frame with exactly the training features."""
    row = dict(form)
    row["Amenities"] = ", ".join(sorted(form["Amenities"]))
    row["Price_per_SqFt"] = round(form["Price_in_Lakhs"] * config.LAKH / form["Size_in_SqFt"], 2)
    row["Age_of_Property"] = config.REFERENCE_YEAR - form["Year_Built"]
    return engineer(pd.DataFrame([row]), market_reference())


@dataclass
class Prediction:
    probability: float
    threshold: float
    is_good: bool
    future_price: float
    future_low: float
    future_high: float
    implied_growth_pct: float
    contributions: pd.Series | None  # log-odds pushed by each feature, grouped to source columns


def predict(prop: pd.DataFrame) -> Prediction:
    clf, reg = models()
    X = model_input(prop)
    probability = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    q = evaluation()["regression"]["log_residual_quantiles"]
    price = float(prop["Price_in_Lakhs"].iloc[0])
    threshold = decision_threshold()
    return Prediction(
        probability=probability, threshold=threshold, is_good=probability >= threshold, future_price=future,
        future_low=future * np.exp(q["p05"]), future_high=future * np.exp(q["p95"]),
        implied_growth_pct=((future / price) ** (1 / config.FORECAST_YEARS) - 1) * 100,
        contributions=contributions(clf, X),
    )


def contributions(clf, X: pd.DataFrame) -> pd.Series | None:
    """Per-feature contributions (SHAP values, in log-odds) from XGBoost's built-in TreeSHAP.
    Returns None if the champion classifier is not an XGBoost model."""
    model = clf.named_steps["model"]
    if not hasattr(model, "get_booster"):
        return None
    import xgboost as xgb

    pre = clf.named_steps["preprocess"]
    values = model.get_booster().predict(xgb.DMatrix(pre.transform(X)), pred_contribs=True)[0][:-1]
    names = pre.get_feature_names_out()
    source = [n if n in config.NUMERIC_FEATURES else
              next((c for c in config.CATEGORICAL_FEATURES if n.startswith(c + "_")), n) for n in names]
    grouped = pd.Series(values, index=source).groupby(level=0).sum()
    return grouped.reindex(grouped.abs().sort_values(ascending=False).index)


def similar_listings(df: pd.DataFrame, prop: pd.DataFrame) -> pd.DataFrame:
    p = prop.iloc[0]
    return df[(df["City"] == p["City"]) & (df["Property_Type"] == p["Property_Type"]) & (df["BHK"] == p["BHK"])]
