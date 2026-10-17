"""Build the two targets from the documented domain rules in config.py."""
import numpy as np
import pandas as pd

import config


def expected_growth_rate(df: pd.DataFrame) -> pd.Series:
    """Annual appreciation (%) the property's features justify, before any market shock."""
    adj = config.GROWTH_ADJUSTMENTS
    rate = (
        config.GROWTH_BASE_RATE
        + df["Public_Transport_Accessibility"].map(adj["transport"])
        + df["Availability_Status"].map(adj["availability"])
        + df["Property_Type"].map(adj["property_type"])
        + df["City_Tier"].map(adj["city_tier"])
        + df["Security"].map(adj["security"])
        + df["Parking_Space"].map(adj["parking"])
        + config.GROWTH_PER_AMENITY * df["Amenity_Count"]
        + config.GROWTH_PER_SOCIAL_FACILITY * (df["Nearby_Schools"] + df["Nearby_Hospitals"])
        + config.GROWTH_PER_YEAR_OF_AGE * (df["Age_of_Property"] - config.GROWTH_AGE_PIVOT_YEARS)
    )
    return rate.astype(float)


def add_targets(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    df["Expected_Growth_Rate"] = expected_growth_rate(df).round(3)
    shock = rng.normal(0, config.GROWTH_NOISE_STD, len(df))
    df["Annual_Growth_Rate"] = (df["Expected_Growth_Rate"] + shock).clip(*config.GROWTH_RATE_BOUNDS).round(3)

    multiplier = (1 + df["Annual_Growth_Rate"] / 100) ** config.FORECAST_YEARS
    df[config.REGRESSION_TARGET] = (df["Price_in_Lakhs"] * multiplier).round(2)

    df[config.CLASSIFICATION_TARGET] = (
        (df["Annual_Growth_Rate"] >= config.GOOD_INVESTMENT_MIN_GROWTH)
        & (df["PPSF_vs_City_Median"] <= config.GOOD_INVESTMENT_MAX_PPSF_RATIO)
    ).astype(int)
    return df
