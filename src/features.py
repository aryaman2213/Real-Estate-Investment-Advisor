"""Feature engineering. Shared by the training pipeline and the app, so a property typed into
the app gets exactly the same features the models were trained on."""
import json

import numpy as np
import pandas as pd

import config

TRANSPORT_SCORE = {"Low": 1, "Medium": 2, "High": 3}
FURNISHED_LEVEL = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}


def city_median_ppsf(df: pd.DataFrame) -> dict:
    """Market reference: median price per sq ft for each city."""
    return {city: round(float(v), 2) for city, v in df.groupby("City")["Price_per_SqFt"].median().items()}


def save_market_reference(df: pd.DataFrame, path=config.MARKET_REFERENCE_PATH) -> dict:
    reference = {"city_median_ppsf": city_median_ppsf(df)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2))
    return reference


def load_market_reference(path=config.MARKET_REFERENCE_PATH) -> dict:
    return json.loads(path.read_text())


def engineer(df: pd.DataFrame, market_reference: dict) -> pd.DataFrame:
    """Add model features. Needs Price_per_SqFt in rupees (the cleaning step guarantees this)."""
    df = df.copy()
    medians = market_reference["city_median_ppsf"]

    df["Log_Price"] = np.log(df["Price_in_Lakhs"])
    df["PPSF_vs_City_Median"] = (df["Price_per_SqFt"] / df["City"].map(medians)).round(4)
    df["City_Tier"] = np.where(df["City"].isin(config.TIER1_CITIES), 1, 2)

    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"]).round(4)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_SCORE)
    df["Furnished_Level"] = df["Furnished_Status"].map(FURNISHED_LEVEL)
    df["Has_Parking"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Has_Security"] = (df["Security"] == "Yes").astype(int)
    df["Is_Ready_To_Move"] = (df["Availability_Status"] == "Ready to Move").astype(int)

    amenity_sets = df["Amenities"].fillna("").map(lambda s: {a.strip() for a in s.split(",") if a.strip()})
    for amenity in config.AMENITY_LIST:
        df[f"Has_{amenity}"] = amenity_sets.map(lambda s, a=amenity: int(a in s))
    df["Amenity_Count"] = amenity_sets.map(len)
    # Amenities per 1,000 sq ft: how much shared facility a buyer gets for the space they pay for.
    df["Amenity_Density_Score"] = (df["Amenity_Count"] / df["Size_in_SqFt"] * 1000).round(4)

    # 0-100 score: schools and hospitals (each 1-10) plus transport access, weighted equally.
    df["Infrastructure_Score"] = (
        100 * (df["Nearby_Schools"] / 10 + df["Nearby_Hospitals"] / 10 + (df["Transport_Score"] - 1) / 2) / 3
    ).round(2)

    # Readable buckets for EDA and the app (not model inputs).
    df["Age_Category"] = pd.cut(
        df["Age_of_Property"], bins=[0, 5, 15, 25, 100], labels=["New (0-5)", "6-15", "16-25", "25+"]
    ).astype("string")
    df["Price_Category"] = pd.cut(
        df["Price_in_Lakhs"], bins=[0, 100, 250, 400, np.inf],
        labels=["Budget (<1 Cr)", "Mid (1-2.5 Cr)", "Premium (2.5-4 Cr)", "Luxury (4 Cr+)"],
    ).astype("string")
    return df
