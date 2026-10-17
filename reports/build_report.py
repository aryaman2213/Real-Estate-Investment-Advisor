"""Build the project report (HTML -> PDF).

Numbers come from models/model_metrics.json and data/processed/cleaning_report.json, written answers from
notebooks/eda.ipynb, charts from reports/figures/ and reports/screenshots/, so the report always matches the
analysis. The PDF is printed by headless Google Chrome through Playwright.

    python reports/build_report.py --live-url https://<your-app>.streamlit.app
"""
import argparse
import base64
import html
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import markdown
import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

REPORTS = ROOT / "reports"
PDF_PATH = REPORTS / "Real_Estate_Investment_Advisor_Report.pdf"
AUTHOR = "Aryaman Bhatnagar"
GITHUB_URL = "https://github.com/aryaman2213/real-estate-investment-advisor"


def md(text: str) -> str:
    text = re.sub(r"([^\n])\n(?=(\d+\.|-) )", r"\1\n\n", text)
    return markdown.markdown(text, extensions=["tables"])


def img(path: Path, caption: str, max_width=1400, quality=82) -> str:
    """Embed an image as a compressed JPEG data URI."""
    im = Image.open(path).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img src="data:image/jpeg;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def fig(name: str, caption: str) -> str:
    return img(config.FIGURES_DIR / f"{name}.png", caption, max_width=1100)


def notebook_answers() -> dict:
    """{'Q1': (question, answer html), ...} from the executed notebook."""
    nb = nbformat.read(ROOT / "notebooks" / "eda.ipynb", as_version=4)
    answers, current = {}, None
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        heading = re.match(r"### (Q\d+)\. (.+)", cell.source)
        if heading:
            current = heading.groups()
        elif cell.source.startswith("**Answer.**") and current:
            answers[current[0]] = (current[1], md(cell.source.removeprefix("**Answer.**").strip()))
            current = None
    return answers


def metrics_table(models: dict, champion: str, cols: list[tuple[str, str, str]]) -> str:
    head = "<th>Model</th>" + "".join(f"<th>{label}</th>" for _, label, _ in cols)
    rows = []
    for name, m in models.items():
        cls = ' class="champ"' if name == champion else ""
        cells = "".join(f"<td>{format(m[key], fmt) if m.get(key) is not None else '–'}</td>" for key, _, fmt in cols)
        rows.append(f"<tr{cls}><td>{'🏆 ' if name == champion else ''}{html.escape(name)}</td>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: Inter, -apple-system, 'Segoe UI', sans-serif; color: #0f172a; font-size: 10.2pt; line-height: 1.5; }
h1 { font-size: 26pt; margin: 0 0 6pt 0; letter-spacing: -0.02em; }
h2 { font-size: 15.5pt; margin: 20pt 0 6pt 0; padding-bottom: 4pt; border-bottom: 2px solid #e0e7ff; color: #312e81; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 3pt 0; break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #e2e8f0; padding: 4pt 6pt; text-align: left; }
th { background: #eef2ff; }
tr.champ td { background: #ecfdf5; font-weight: 600; }
code { background: #f1f5f9; padding: 0 3pt; border-radius: 3pt; font-size: 8.8pt; }
figure { margin: 6pt 0 10pt 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 225mm; border: 1px solid #e2e8f0; border-radius: 6px; }
figcaption { color: #64748b; font-size: 8.5pt; margin-top: 3pt; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.eyebrow { color: #4f46e5; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; font-size: 9pt; }
.sub { color: #475569; font-size: 12.5pt; max-width: 150mm; }
.meta td { border: none; padding: 2pt 10pt 2pt 0; font-size: 10pt; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin: 8pt 0; break-inside: avoid; }
.kpi { border: 1px solid #e2e8f0; border-radius: 8px; padding: 6pt 8pt; }
.kpi b { display: block; font-size: 14pt; } .kpi span { color: #64748b; font-size: 8.5pt; }
.qa { break-inside: avoid; margin-bottom: 6pt; } .qa .q { font-weight: 650; }
.box { background: #f8fafc; border-left: 4px solid #4f46e5; padding: 6pt 10pt; margin: 8pt 0; break-inside: avoid; }
.page { break-before: page; }
"""


def build_html(live_url: str | None) -> str:
    m = json.loads(config.METRICS_PATH.read_text())
    rep = json.loads(config.CLEANING_REPORT_PATH.read_text())
    answers = notebook_answers()
    clf, reg = m["classification"], m["regression"]
    cc, rc = clf["models"][clf["champion"]], reg["models"][reg["champion"]]
    rb = reg["models"]["Baseline (fixed 8% rate)"]
    cm = json.loads(config.EVALUATION_PATH.read_text())["classification"]["confusion_matrices"][clf["champion"]]
    adj = config.GROWTH_ADJUSTMENTS

    def qa(ids):
        return "".join(f'<div class="qa"><p class="q">{q}. {html.escape(answers[q][0])}</p>{answers[q][1]}</div>'
                       for q in ids if q in answers)

    links = f'<tr><td>GitHub</td><td><a href="{GITHUB_URL}">{GITHUB_URL}</a></td></tr>'
    if live_url:
        links += f'<tr><td>Live app</td><td><a href="{live_url}">{live_url}</a></td></tr>'
    shots = REPORTS / "screenshots"

    body = f"""
<div class="cover">
  <div class="eyebrow">Capstone project · Real estate / investment analytics</div>
  <h1>Real Estate Investment Advisor</h1>
  <p class="sub">Predicting property profitability and future value: a classifier for <b>Good Investment</b>
  and a regressor for the <b>price in 5 years</b>, tracked with MLflow and served through a Streamlit app.</p>
  <table class="meta">
    <tr><td>Author</td><td>{AUTHOR}</td></tr>
    <tr><td>Date</td><td>{date.today():%d %B %Y}</td></tr>
    {links}
    <tr><td>Stack</td><td>Python, pandas, scikit-learn, XGBoost, MLflow, Streamlit, Plotly</td></tr>
  </table>
  <div class="kpis">
    <div class="kpi"><b>250,000</b><span>listings, 42 cities, 20 states</span></div>
    <div class="kpi"><b>{cc['roc_auc']:.3f}</b><span>classifier ROC AUC · {clf['champion']}</span></div>
    <div class="kpi"><b>{rc['rmse']:.1f} L</b><span>regressor RMSE (R² {rc['r2']:.4f}) · {reg['champion']}</span></div>
    <div class="kpi"><b>{len(clf['models']) - 1 + len(reg['models']) - 1}</b><span>models trained and logged in MLflow</span></div>
  </div>
</div>

<h2>1. Problem statement and business use cases</h2>
<p>Build a machine-learning application that helps investors decide (1) whether a property is a <b>Good Investment</b>
(classification) and (2) what it will be worth <b>after 5 years</b> (regression), with experiment tracking in MLflow and
an interactive Streamlit app.</p>
<ul>
<li><b>Investors</b> assess long-term returns before buying.</li>
<li><b>Buyers</b> find high-return properties in developing areas (the app ranks cities and lists the top-scored properties).</li>
<li><b>Real estate platforms</b> automate investment analysis for every listing (the explorer scores all 250,000).</li>
<li><b>Trust</b>: every prediction shows its confidence, a forecast range and the reasons behind it.</li>
</ul>

<h2>2. Data and cleaning</h2>
<p><code>india_housing_prices.csv</code>: {rep['raw_shape'][0]:,} rows × {rep['raw_shape'][1]} columns (location, type, size,
price, age, floors, nearby schools and hospitals, transport, parking, security, amenities, facing, owner, availability).</p>
<table><thead><tr><th>Check</th><th>Finding</th><th>Action</th></tr></thead><tbody>
<tr><td>Missing values</td><td>None</td><td>Median / mode imputation kept in the pipeline as a safety net for app input</td></tr>
<tr><td>Duplicates</td><td>None (also none ignoring ID)</td><td>—</td></tr>
<tr><td>Floor number &gt; total floors</td><td>{rep['floor_above_total_floors']['rows']:,} rows ({rep['floor_above_total_floors']['pct']}%)</td><td>Values swapped back (column mix-up), no rows lost</td></tr>
<tr><td>Price_per_SqFt units</td><td>Stored in lakhs, rounded to 2 decimals</td><td>Recomputed exactly in rupees</td></tr>
<tr><td>Outliers (IQR)</td><td>Price 0, size 0, price per sq ft {rep['outliers_iqr']['Price_per_SqFt']['outliers']:,}</td><td>Genuine small, expensive homes: flagged, kept; log price for linear models</td></tr>
<tr><td>Locality</td><td>Each of 500 values appears in all 42 cities</td><td>Excluded from models (no location information)</td></tr>
<tr><td>State / Year_Built</td><td>Determined by City / Age</td><td>Excluded as redundant</td></tr>
</tbody></table>
<p><b>Key EDA finding:</b> the listings are synthetic. Every raw column is uniformly distributed and independent of
every other (size vs price r ≈ 0.00; average price per city varies by only 3%).</p>

<h3>Feature engineering</h3>
<p><code>Price_per_SqFt</code> (₹), <code>Log_Price</code>, <code>PPSF_vs_City_Median</code>, <code>City_Tier</code> (8 X-class metros vs the rest),
<code>Infrastructure_Score</code> (0–100: schools, hospitals, transport), <code>Amenity_Count</code>,
<code>Amenity_Density_Score</code> (amenities per 1,000 sq ft), five <code>Has_&lt;amenity&gt;</code> flags,
<code>Transport_Score</code>, <code>Furnished_Level</code>, <code>Floor_Ratio</code>, and binary parking / security / ready-to-move flags.</p>

<h2>3. Target definitions</h2>
<p>The dataset has no sale history, so both targets follow documented domain rules, as the brief allows.</p>
<div class="box"><b>Future_Price_5Y</b> = Price × (1 + r)<sup>5</sup>, where the annual appreciation rate r = {config.GROWTH_BASE_RATE}% base
+ feature adjustments + a random market shock (σ = {config.GROWTH_NOISE_STD} pp) for what a listing cannot capture.
The market-wide average is ≈ 8%, the rate in the brief's example.</div>
<table><thead><tr><th>Driver</th><th>Adjustment (pp per year)</th><th>Reason</th></tr></thead><tbody>
<tr><td>Public transport</td><td>Low {adj['transport']['Low']:+}, Medium 0, High {adj['transport']['High']:+}</td><td>Connectivity drives resale demand</td></tr>
<tr><td>Under construction</td><td>{adj['availability']['Under Construction']:+}</td><td>Pre-launch pricing; value rises on completion</td></tr>
<tr><td>Property type</td><td>Villa {adj['property_type']['Villa']:+}, Independent house {adj['property_type']['Independent House']:+}</td><td>Land appreciates; buildings age</td></tr>
<tr><td>Tier 2 city</td><td>{adj['city_tier'][2]:+}</td><td>Developing markets grow faster from a lower base</td></tr>
<tr><td>Security, parking</td><td>{adj['security']['Yes']:+} each</td><td>Resale premium</td></tr>
<tr><td>Amenities; schools + hospitals</td><td>{config.GROWTH_PER_AMENITY:+} each; {config.GROWTH_PER_SOCIAL_FACILITY:+} each</td><td>Facilities and social infrastructure</td></tr>
<tr><td>Age</td><td>{config.GROWTH_PER_YEAR_OF_AGE:+} per year of age above {config.GROWTH_AGE_PIVOT_YEARS} (newer homes gain the same per year below it)</td><td>Depreciation of older buildings</td></tr>
</tbody></table>
<div class="box"><b>Good_Investment</b> = 1 when annual appreciation ≥ <b>{config.GOOD_INVESTMENT_MIN_GROWTH:.0f}%</b> <b>and</b>
price per sq ft ≤ the <b>city median</b>. This combines the brief's three suggested rules (appreciation above a threshold,
cheaper than the median, multi-factor). Result: {rep['good_investment_rate_pct']}% of listings are Good Investments.</div>

<h2>4. Exploratory data analysis</h2>
<h3>Price and size (Q1–Q5)</h3>
{qa(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])}
{fig('q05_outliers', 'Q5. Outliers: only price per sq ft has them (IQR rule)')}
<h3>Location (Q6–Q10)</h3>
{qa(['Q6', 'Q7', 'Q8', 'Q9', 'Q10'])}
<h3>Relationships (Q11–Q15)</h3>
{qa(['Q11', 'Q12', 'Q13', 'Q14', 'Q15'])}
{fig('q11_correlation', 'Q11. Raw features are uncorrelated; the structure lives in the engineered features and targets')}
<h3>Investment, amenities and ownership (Q16–Q20)</h3>
{qa(['Q16', 'Q17', 'Q18', 'Q19', 'Q20'])}
{fig('q20_transport', 'Q20. Transport does not change price per sq ft but strongly changes investment potential')}
<h3>Investment-focused analyses</h3>
{fig('s1_investment_by_city', 'Share of Good Investments by city: Tier 2 (green) ahead of Tier 1 metros (blue)')}
{fig('s3_infrastructure_vs_resale', 'Infrastructure score vs expected 5-year resale gain')}
<p>The dataset has <b>no crime-rate column</b>; the closest proxy, security, raises the Good Investment share from 25.3% to 28.8%.</p>
<h3>Statistical tests</h3>
<ul>
<li><b>Chi-square test of independence</b> (categorical features vs Good_Investment) with Cramér's V for effect size: transport
(V ≈ 0.30), availability (0.12), city tier (0.09) and property type (0.07) are significant; owner type, furnishing and
facing are not (p &gt; 0.7).</li>
<li><b>Welch t-test</b>: under-construction properties grow significantly faster than ready-to-move ones.</li>
<li><b>One-way ANOVA</b>: growth differs significantly across transport levels.</li>
<li><b>Kruskal-Wallis</b> (non-parametric, skewed data): price per sq ft does <b>not</b> differ across the 42 cities (p ≈ 0.66).</li>
<li><b>Pearson</b>: size vs price r ≈ 0.00 (p ≈ 0.2).</li>
</ul>
<p>With 250,000 rows, tiny differences become "significant", so effect sizes were read alongside p-values (e.g. property
type changes price per sq ft by ~1%: significant but not meaningful).</p>

<h2 class="page">5. Model development</h2>
<ul>
<li>One <b>stratified 70/15/15 split</b>, shared by both tasks: {m['train_rows']:,} train / {m['val_rows']:,} validation /
{m['test_rows']:,} test. Models are fitted on train; <b>validation</b> picks the champions, each classifier's decision threshold
and the forecast range; the <b>test</b> set is scored once and is the only set reported.</li>
<li>A scikit-learn <b>Pipeline</b>: median/mode imputation → StandardScaler (numeric) + OneHotEncoder (categorical) → model,
so the app applies exactly the same preprocessing.</li>
<li><b>Class imbalance</b> ({100 * (1 - clf['positive_rate']):.0f}:{100 * clf['positive_rate']:.0f}): stratified split, <code>class_weight="balanced"</code>,
XGBoost <code>scale_pos_weight</code> = {clf['pos_weight']:.2f}, and each classifier's probability cut-off tuned on validation to
maximise F1. Models are compared on validation ROC AUC with F1 as tie-break (accuracy is misleading: the majority-class
baseline already scores {100 * clf['models']['Baseline (majority class)']['accuracy']:.0f}%).</li>
<li><b>Regression in log space</b>: price × (1 + r)<sup>5</sup> is multiplicative; log turns it additive, so every model fits it easily.</li>
<li><b>Hyperparameter tuning</b>: XGBoost, randomised search ({config.TUNING_ITERATIONS} candidates, {config.CV_FOLDS}-fold CV on a
{config.TUNING_SAMPLE_SIZE:,}-row sample), logged as its own run.</li>
</ul>
<h3>Classification: Good Investment</h3>
{metrics_table(clf['models'], clf['champion'], [('val_roc_auc', 'Val AUC', '.4f'), ('roc_auc', 'Test AUC', '.4f'), ('f1', 'F1', '.4f'),
                                               ('accuracy', 'Accuracy', '.4f'), ('precision', 'Precision', '.4f'),
                                               ('recall', 'Recall', '.4f'), ('threshold', 'Threshold', '.2f')])}
<p><b>Champion: {clf['champion']}</b>: best validation ROC AUC; on the test set ROC AUC {cc['roc_auc']:.3f}, F1 {cc['f1']:.3f},
accuracy {100 * cc['accuracy']:.1f}%, precision {100 * cc['precision']:.1f}%, recall {100 * cc['recall']:.1f}% at a decision threshold of
{clf['threshold']:.2f}. Confusion matrix (test): {cm[0][0]:,} true negatives, {cm[0][1]:,} false positives, {cm[1][0]:,} false negatives,
{cm[1][1]:,} true positives. Boosted trees lead because the label is an AND of two thresholds, which trees represent
naturally; the remaining errors sit near the 8% growth line, where the random market shock decides the label.</p>
<h3>Regression: price in 5 years</h3>
{metrics_table(reg['models'], reg['champion'], [('val_rmse', 'Val RMSE', '.3f'), ('rmse', 'Test RMSE (L)', '.3f'),
                                               ('mae', 'MAE (L)', '.3f'), ('mse', 'MSE', '.2f'),
                                               ('r2', 'R²', '.5f'), ('mape_pct', 'MAPE %', '.2f')])}
<p><b>Champion: {reg['champion']}</b>, RMSE {rc['rmse']:.2f} L ({100 * (1 - rc['rmse'] / rb['rmse']):.0f}% lower than the brief's fixed-8% formula)
and MAPE {rc['mape_pct']:.2f}%. In log space future price is a linear function of the features, so the simplest
model wins and is also the fastest and most explainable. R² is high for every model because today's price explains
most of the future price; RMSE against the baseline shows the real gain. The remaining error is the random market shock,
which no model can predict. The app's 90% forecast range is calibrated on validation residuals and contains the true
value for {100 * reg['interval_90_test_coverage']:.1f}% of test listings.</p>

<h2>6. MLflow tracking and registry</h2>
<ul>
<li>Two experiments: <code>{config.EXPERIMENT_CLASSIFICATION}</code> and <code>{config.EXPERIMENT_REGRESSION}</code>.</li>
<li>Every run logs hyperparameters, all metrics, a confusion matrix or actual-vs-predicted plot, a feature-importance chart
and JSON, and the fitted pipeline with its input signature.</li>
<li>Registry: <code>{config.REGISTERED_CLASSIFIER}</code> v{clf['registry_version']} and <code>{config.REGISTERED_REGRESSOR}</code>
v{reg['registry_version']}, each with the alias <code>@{config.CHAMPION_ALIAS}</code>. The app loads these champions (exported to <code>models/</code>).</li>
<li><code>python evaluate.py</code> reloads both champions from the registry (<code>models:/…@champion</code>), re-scores the test set, prints the
classification report and confusion matrix, and asserts that the numbers and the app's model files match exactly.</li>
<li><code>python mlflow_ui.py</code> opens the MLflow UI; it first re-points MLflow's stored artifact paths so a fresh clone works anywhere.</li>
</ul>

<h2>7. Streamlit application</h2>
<p>Six pages: <b>Investment advisor</b> (form → verdict with confidence, 5-year forecast with a 90% range, the two rule checks,
SHAP explanation, similar listings), <b>Property explorer</b> (all 250,000 listings scored; filter by area, price, BHK,
availability; search; sort; CSV download), <b>Location insights</b> (map heatmap, state × type heatmap, city ranking, price
trend), <b>Market analysis</b>, <b>Model performance</b> and <b>About</b>.</p>
{img(shots / '01_advisor_good_investment.png', 'Investment advisor: a Good Investment with its forecast, checks and explanation')}
{img(shots / '02_advisor_not_good.png', 'The same property at ₹4.2 Cr fails the price check, and the verdict flips')}
{img(shots / '03_property_explorer.png', 'Property explorer: every listing scored by both models')}
{img(shots / '05_location_insights.png', 'Location insights: map heatmap, city ranking, state × property type, trends')}
{img(shots / '07_models_classification.png', 'Model performance: classification comparison, confusion matrix, ROC, importance')}
{img(shots / '08_models_regression.png', 'Model performance: regression comparison and actual vs predicted')}

<h2>8. Insights and business recommendations</h2>
<ol>
<li><b>Connectivity first.</b> High-transport properties are Good Investments 43% of the time vs 11% for low transport. Platforms
should surface transport access prominently, and investors should weight it above furnishing or facing.</li>
<li><b>Buy the asset, not the finish.</b> Furnishing, facing and owner type show no relationship with price or returns.</li>
<li><b>Look beyond the metros.</b> Tier 2 cities offer 29% Good Investments vs 19% in the eight metros, matching the
"developing areas" use case.</li>
<li><b>Price per sq ft matters more than price.</b> Larger homes are cheaper per sq ft and pass the value test far more often
(45% Good Investments above 4,000 sq ft vs 7% below 1,000).</li>
<li><b>Under-construction purchases</b> carry higher expected appreciation; pair them with developer due diligence (RERA).</li>
<li><b>Use the model as a screen, not a verdict.</b> It ranks 250,000 listings in seconds; humans review the shortlist.</li>
</ol>
<h3>Limitations and next steps</h3>
<ul>
<li>The listings are synthetic and the targets are rule-based, so the models show how well each algorithm recovers a
documented investment rule, not real market behaviour.</li>
<li>With real transactions (registration data, historical prices), the same pipeline would learn the growth drivers
directly. Useful additions: crime index, RERA status, distance to metro and business districts.</li>
<li>Monitoring: retrain on new data and compare against the registered champion in MLflow before promoting.</li>
</ul>

<h2>9. Deliverables</h2>
<table><tbody>
<tr><td>Cleaned dataset</td><td><code>data/processed/housing_clean.csv</code></td></tr>
<tr><td>Scripts</td><td><code>pipeline.py</code> (data), <code>train.py</code> (training + MLflow), <code>evaluate.py</code> (evaluation),
<code>src/</code> (cleaning, features, targets, EDA, modelling, tracking)</td></tr>
<tr><td>EDA notebook</td><td><code>notebooks/eda.ipynb</code> (all 20 questions + statistical tests)</td></tr>
<tr><td>MLflow logs + registry</td><td><code>mlflow.db</code>, <code>mlruns/</code>, viewed with <code>python mlflow_ui.py</code></td></tr>
<tr><td>Streamlit app</td><td><code>app/streamlit_app.py</code>{f' · <a href="{live_url}">{live_url}</a>' if live_url else ''}</td></tr>
<tr><td>Documentation</td><td><code>README.md</code>, this report, <code>reports/screenshots/</code></td></tr>
</tbody></table>
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-url", default=None, help="Streamlit Cloud URL, once deployed")
    args = parser.parse_args()
    page_html = build_html(args.live_url)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.set_content(page_html, wait_until="load")
        page.pdf(path=str(PDF_PATH), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;color:#94a3b8;width:100%;text-align:center;">'
                                 'Real Estate Investment Advisor · <span class="pageNumber"></span>/<span class="totalPages"></span></div>')
        browser.close()
    print(f"Saved {PDF_PATH.relative_to(ROOT)} ({PDF_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
