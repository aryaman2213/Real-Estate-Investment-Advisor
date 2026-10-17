"""Load the raw CSV and clean it. Every step records what it found, for the cleaning report."""
import numpy as np
import pandas as pd

import config

NUMERIC_COLUMNS = [
    "ID", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built", "Floor_No",
    "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
]
TEXT_COLUMNS = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status",
]


def load_raw(path=config.RAW_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def standardize_text(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Trim whitespace, collapse repeated spaces and apply readable category labels."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    for col, mapping in config.CATEGORY_RENAMES.items():
        df[col] = df[col].replace(mapping)
    # Amenities is a comma-separated list: normalise spacing and order so equal sets compare equal.
    df["Amenities"] = df["Amenities"].map(
        lambda s: ", ".join(sorted(a.strip() for a in s.split(","))) if isinstance(s, str) else s
    )
    report["category_levels"] = {c: int(df[c].nunique()) for c in TEXT_COLUMNS}
    return df


def fix_types(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Coerce numeric columns; anything unparseable becomes NaN and is imputed next."""
    df = df.copy()
    before = df[NUMERIC_COLUMNS].isna().sum().sum()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    report["values_made_missing_by_type_fix"] = int(df[NUMERIC_COLUMNS].isna().sum().sum() - before)
    return df


def handle_missing(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Median for numeric columns (robust to skew), mode for categorical columns."""
    df = df.copy()
    missing = df.isna().sum()
    report["missing_before"] = {c: int(n) for c, n in missing.items() if n}
    for col in NUMERIC_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in TEXT_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    report["missing_after"] = int(df.isna().sum().sum())
    return df


def remove_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Exact duplicates, then the same listing under a different ID."""
    exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    same_listing = int(df.drop(columns="ID").duplicated().sum())
    df = df.loc[~df.drop(columns="ID").duplicated()]
    report["duplicates_removed"] = {"exact": exact, "same_listing_different_id": same_listing}
    return df


def fix_invalid_values(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Repair values that cannot be true, and recompute derived columns at full precision."""
    df = df.copy()
    invalid_floor = df["Floor_No"] > df["Total_Floors"]
    report["floor_above_total_floors"] = {
        "rows": int(invalid_floor.sum()),
        "pct": round(100 * invalid_floor.mean(), 2),
        "action": config.FLOOR_FIX_STRATEGY,
    }
    if config.FLOOR_FIX_STRATEGY == "swap":
        df.loc[invalid_floor, ["Floor_No", "Total_Floors"]] = (
            df.loc[invalid_floor, ["Total_Floors", "Floor_No"]].to_numpy()
        )
    else:  # "clip"
        df.loc[invalid_floor, "Floor_No"] = df.loc[invalid_floor, "Total_Floors"]

    # Age must agree with Year_Built; recompute so the two can never disagree.
    report["age_mismatches"] = int((config.REFERENCE_YEAR - df["Year_Built"] != df["Age_of_Property"]).sum())
    df["Age_of_Property"] = config.REFERENCE_YEAR - df["Year_Built"]

    # The raw Price_per_SqFt is in lakhs and rounded to 2 decimals (0.10 = anything from
    # Rs 9,500 to Rs 10,499 per sq ft). Recompute it exactly, in rupees.
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * config.LAKH / df["Size_in_SqFt"]).round(2)

    impossible = (df["Price_in_Lakhs"] <= 0) | (df["Size_in_SqFt"] <= 0) | (df["BHK"] <= 0)
    report["impossible_rows_dropped"] = int(impossible.sum())
    return df.loc[~impossible]


def detect_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """IQR fences on the money and size columns.

    Outliers are flagged, not removed: price and size are bounded (10-500 lakhs,
    500-5,000 sq ft), so every high price-per-sq-ft value is a real small, expensive home.
    Tree models are unaffected, and the linear models get Log_Price.
    """
    df = df.copy()
    k = config.OUTLIER_IQR_MULTIPLIER
    summary = {}
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (df[col] < low) | (df[col] > high)
        summary[col] = {"lower_fence": round(low, 2), "upper_fence": round(high, 2), "outliers": int(mask.sum())}
        if col == "Price_per_SqFt":
            df["Is_PPSF_Outlier"] = mask.astype(int)
    report["outliers_iqr"] = summary
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"raw_shape": list(df.shape)}
    df = standardize_text(df, report)
    df = fix_types(df, report)
    df = handle_missing(df, report)
    df = remove_duplicates(df, report)
    df = fix_invalid_values(df, report)
    df = detect_outliers(df, report)
    df = df.reset_index(drop=True)
    # Whole-number columns back to integers (imputation can leave them as floats). Prices stay decimal.
    for col in NUMERIC_COLUMNS:
        if col not in ("Price_in_Lakhs", "Price_per_SqFt"):
            df[col] = df[col].round().astype(np.int64)
    report["clean_shape"] = list(df.shape)
    return df, report
