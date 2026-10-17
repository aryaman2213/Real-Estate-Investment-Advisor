"""Central configuration: paths, cleaning rules, target-definition rules and model settings.

Every number that shapes the result lives here, so an assumption can be changed in one place
and the whole pipeline re-run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths -----------------------------------------------------------------
RAW_CSV_PATH = ROOT / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV_PATH = PROCESSED_DIR / "housing_clean.csv"
APP_DATA_PATH = PROCESSED_DIR / "housing_clean.parquet"  # same rows, compact copy for the app
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"
MODELS_DIR = ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "good_investment_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "future_price_regressor.joblib"
MARKET_REFERENCE_PATH = MODELS_DIR / "market_reference.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
EVALUATION_PATH = MODELS_DIR / "evaluation_details.json"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- MLflow ----------------------------------------------------------------
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = ROOT / "mlruns"
EXPERIMENT_CLASSIFICATION = "good_investment_classification"
EXPERIMENT_REGRESSION = "future_price_regression"
REGISTERED_CLASSIFIER = "RealEstate_GoodInvestment_Classifier"
REGISTERED_REGRESSOR = "RealEstate_FuturePrice5Y_Regressor"
CHAMPION_ALIAS = "champion"

# --- Dataset facts -----------------------------------------------------------
# Age_of_Property in the raw file equals 2025 - Year_Built for every row.
REFERENCE_YEAR = 2025
LAKH = 100_000
AMENITY_LIST = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]

# Raw labels -> readable labels (the raw file uses underscores in some categories).
CATEGORY_RENAMES = {
    "Availability_Status": {"Under_Construction": "Under Construction", "Ready_to_Move": "Ready to Move"},
}

# --- Cleaning rules ----------------------------------------------------------
OUTLIER_IQR_MULTIPLIER = 1.5
# Rows where Floor_No > Total_Floors are physically impossible. The two values are
# swapped (most likely a column mix-up at entry) instead of dropping ~47% of the data.
FLOOR_FIX_STRATEGY = "swap"

# --- City tiers --------------------------------------------------------------
# Tier 1 = the eight X-class metros under the Government of India HRA classification
# (Dwarka is part of Delhi). Every other city is treated as Tier 2, a developing market.
TIER1_CITIES = {
    "Ahmedabad", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Mumbai", "New Delhi", "Pune", "Dwarka",
}

# Approximate city coordinates, used only to draw the map in the app.
CITY_COORDINATES = {
    "Ahmedabad": (23.02, 72.57), "Amritsar": (31.63, 74.87), "Bangalore": (12.97, 77.59),
    "Bhopal": (23.26, 77.41), "Bhubaneswar": (20.30, 85.82), "Bilaspur": (22.08, 82.14),
    "Chennai": (13.08, 80.27), "Coimbatore": (11.02, 76.96), "Cuttack": (20.46, 85.88),
    "Dehradun": (30.32, 78.03), "Durgapur": (23.52, 87.31), "Dwarka": (28.59, 77.05),
    "Faridabad": (28.41, 77.32), "Gaya": (24.79, 85.00), "Gurgaon": (28.46, 77.03),
    "Guwahati": (26.14, 91.74), "Haridwar": (29.95, 78.16), "Hyderabad": (17.39, 78.49),
    "Indore": (22.72, 75.86), "Jaipur": (26.91, 75.79), "Jamshedpur": (22.80, 86.20),
    "Jodhpur": (26.24, 73.02), "Kochi": (9.93, 76.27), "Kolkata": (22.57, 88.36),
    "Lucknow": (26.85, 80.95), "Ludhiana": (30.90, 75.86), "Mangalore": (12.91, 74.86),
    "Mumbai": (19.08, 72.88), "Mysore": (12.30, 76.64), "Nagpur": (21.15, 79.09),
    "New Delhi": (28.61, 77.21), "Noida": (28.54, 77.39), "Patna": (25.59, 85.14),
    "Pune": (18.52, 73.86), "Raipur": (21.25, 81.63), "Ranchi": (23.34, 85.31),
    "Silchar": (24.83, 92.80), "Surat": (21.17, 72.83), "Trivandrum": (8.52, 76.94),
    "Vijayawada": (16.51, 80.65), "Vishakhapatnam": (17.69, 83.22), "Warangal": (17.97, 79.59),
}

# --- Target definitions ------------------------------------------------------
# The dataset has no sale history, so both targets are built from documented domain rules
# (the brief allows this: "Fixed Rate", "By Location/Type" and "Feature-based" growth).
#
# Annual appreciation rate (% per year) = BASE + sum of feature adjustments + market shock.
# Each adjustment is a percentage-point change with a plain real-estate reason.
GROWTH_BASE_RATE = 5.5  # chosen so the market-wide average is ~8%, the brief's example rate
GROWTH_ADJUSTMENTS = {
    # Connectivity is the strongest driver of resale demand.
    "transport": {"Low": -1.0, "Medium": 0.0, "High": 1.0},
    # Buying under construction locks in pre-launch pricing; the asset gains on completion.
    "availability": {"Under Construction": 0.75, "Ready to Move": 0.0},
    # Land-heavy property appreciates faster than apartments, whose building ages.
    "property_type": {"Villa": 0.5, "Independent House": 0.25, "Apartment": 0.0},
    # Developing (Tier 2) cities grow faster from a lower base than mature metros.
    "city_tier": {1: 0.0, 2: 0.75},
    "security": {"Yes": 0.25, "No": 0.0},
    "parking": {"Yes": 0.25, "No": 0.0},
}
GROWTH_PER_AMENITY = 0.25            # pp for each amenity (1-5 in the data)
GROWTH_PER_SOCIAL_FACILITY = 0.05    # pp for each nearby school or hospital (2-20 in total)
GROWTH_AGE_PIVOT_YEARS = 15          # buildings older than this appreciate slower...
GROWTH_PER_YEAR_OF_AGE = -0.04       # ...by 0.04 pp per year (and newer ones faster)
# Market shock: everything the listing cannot tell us (policy, interest rates, local news).
GROWTH_NOISE_STD = 0.75
GROWTH_RATE_BOUNDS = (2.0, 15.0)
FORECAST_YEARS = 5
RANDOM_SEED = 42

# Good Investment = appreciates at least as fast as the benchmark AND is not overpriced
# for its city. Both conditions come from the brief ("appreciation rate > threshold",
# "price per sq ft cheaper than median").
GOOD_INVESTMENT_MIN_GROWTH = 8.0     # the brief's example fixed rate r = 8%
GOOD_INVESTMENT_MAX_PPSF_RATIO = 1.0  # price per sq ft <= city median

# --- Model features ------------------------------------------------------------
# Locality is dropped: every one of its 500 values appears in all 42 cities, so it carries
# no location information. State is dropped because City determines it.
CATEGORICAL_FEATURES = ["City", "Property_Type", "Facing", "Owner_Type"]
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Log_Price", "Price_per_SqFt", "PPSF_vs_City_Median",
    "Age_of_Property", "Floor_No", "Total_Floors", "Floor_Ratio",
    "Nearby_Schools", "Nearby_Hospitals", "Transport_Score", "Infrastructure_Score",
    "Furnished_Level", "Has_Parking", "Has_Security", "Is_Ready_To_Move", "City_Tier",
    "Amenity_Count", "Amenity_Density_Score",
] + [f"Has_{a}" for a in AMENITY_LIST]

CLASSIFICATION_TARGET = "Good_Investment"
REGRESSION_TARGET = "Future_Price_5Y"

VALID_SIZE = 0.15  # model selection, decision threshold, forecast-range calibration
TEST_SIZE = 0.15   # scored once, reported everywhere
TUNING_SAMPLE_SIZE = 40_000
TUNING_ITERATIONS = 12
CV_FOLDS = 3
