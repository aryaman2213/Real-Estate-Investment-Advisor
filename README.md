# Real Estate Investment Advisor

Predicting property profitability and future value for 250,000 Indian listings:
- **Classification:** is the property a *Good Investment*? Champion: tuned XGBoost (ROC AUC 0.971, F1 0.827).
- **Regression:** what is the price after 5 years? Champion: Linear Regression on a log target (RMSE 15.0 lakhs, R² 0.995). That error is 45% lower than the fixed-8% formula.

Experiments are tracked in MLflow, and the best model for each task is registered with the alias `@champion`. A six-page Streamlit app serves both models: an advisor form, a property explorer, location insights, market analysis, model performance and methodology.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py       # clean + engineer features + build targets -> data/processed/
python train.py          # 14 models + baselines, MLflow logging, registry, export to models/
streamlit run app/streamlit_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse experiments
```

## Project structure
| Path | What it does |
|---|---|
| `config.py` | Every path, cleaning rule, target rule and model setting |
| `src/cleaning.py` | Cleaning: types, missing values, duplicates, floor swap fix, price per sq ft recomputed in ₹, IQR outliers |
| `src/features.py` | Feature engineering, shared by training and the app |
| `src/targets.py` | Growth rate → `Future_Price_5Y` and `Good_Investment` |
| `src/eda.py` | One function per EDA question, plus the statistical tests |
| `src/modeling.py`, `src/tracking.py` | Model zoo, pipelines, metrics; MLflow logging and registry |
| `notebooks/eda.ipynb` | The 20 EDA questions answered, plus the statistical tests and insights |
| `app/` | The Streamlit app |
| `reports/` | PDF report, figures, screenshots, and the scripts that build them |

## Target definitions (assumptions)
The data has no sale history, so the targets follow documented domain rules, which the brief allows:
- **Future_Price_5Y** = Price × (1 + r)^5. The rate r is 5.5% base plus adjustments for transport (±1), under construction (+0.75), villa or independent house (+0.5 / +0.25), Tier 2 city (+0.75), security and parking (+0.25 each), amenities (+0.25 each), schools and hospitals (+0.05 each) and age (−0.04 per year beyond 15). A random market shock (σ 0.75 pp) is added on top. The market-wide average is about 8%.
- **Good_Investment** = annual growth ≥ 8% **and** price per sq ft ≤ the city median. 27% of listings qualify.

## Data findings
- The data is synthetic: every column is uniform and independent (size vs price r ≈ 0).
- `Locality` repeats across all 42 cities, so it is dropped from the models.
- Floor number is above total floors in 46.5% of rows; those two values were swapped back.
- `Price_per_SqFt` was stored in lakhs, rounded to 2 decimals; it is recomputed exactly in ₹.

The full methodology and findings are in `reports/Real_Estate_Investment_Advisor_Report.pdf`.
