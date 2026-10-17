"""Step 1 of the project: raw CSV -> cleaned, feature-engineered, labelled dataset.

Run:  python pipeline.py
Then: python train.py
"""
import json

import config
from src.cleaning import clean, load_raw
from src.features import engineer, save_market_reference
from src.targets import add_targets


def main() -> None:
    raw = load_raw()
    df, report = clean(raw)
    reference = save_market_reference(df)
    df = add_targets(engineer(df, reference))

    report["engineered_shape"] = list(df.shape)
    report["good_investment_rate_pct"] = round(100 * df[config.CLASSIFICATION_TARGET].mean(), 2)
    report["growth_rate_summary"] = df["Annual_Growth_Rate"].describe().round(3).to_dict()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CSV_PATH, index=False)
    df.to_parquet(config.APP_DATA_PATH, index=False)
    config.CLEANING_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report, indent=2, default=str))
    print(f"Saved {len(df):,} rows -> {config.CLEAN_CSV_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
