"""EDA: one function per question in the brief. Each returns a summary table and saves its chart
to reports/figures/, so the notebook, the report and the app all use the same numbers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

PALETTE = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#e11d48", "#8b5cf6"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)


def load(path=config.APP_DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)  # saved to disk; the notebook shows the PNG
    return fig


def _describe(s: pd.Series) -> pd.DataFrame:
    d = s.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    d["skew"] = s.skew()
    return pd.DataFrame(d, index=[s.name]).round(2)


# --- 1-5: Price & size ------------------------------------------------------------
def q01_price_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax, color=PALETTE[0])
    ax.set(title="Q1. Distribution of property price", xlabel="Price (lakhs)", ylabel="Properties")
    _save(fig, "q01_price_distribution")
    return _describe(df["Price_in_Lakhs"]), fig


def q02_size_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax, color=PALETTE[1])
    ax.set(title="Q2. Distribution of property size", xlabel="Size (sq ft)", ylabel="Properties")
    _save(fig, "q02_size_distribution")
    return _describe(df["Size_in_SqFt"]), fig


def q03_ppsf_by_property_type(df):
    table = df.groupby("Property_Type")["Price_per_SqFt"].agg(["count", "mean", "median"]).round(0)
    groups = [g["Price_per_SqFt"].to_numpy() for _, g in df.groupby("Property_Type")]
    table.attrs["kruskal_p"] = stats.kruskal(*groups).pvalue
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", ax=ax, showfliers=False, hue="Property_Type", legend=False)
    ax.set(title="Q3. Price per sq ft by property type (outliers hidden)", xlabel="", ylabel="Rs per sq ft")
    _save(fig, "q03_ppsf_by_property_type")
    return table, fig


def q04_size_vs_price(df):
    r, p = stats.pearsonr(df["Size_in_SqFt"], df["Price_in_Lakhs"])
    rho, _ = stats.spearmanr(df["Size_in_SqFt"], df["Price_in_Lakhs"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = ax.hexbin(df["Size_in_SqFt"], df["Price_in_Lakhs"], gridsize=40, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Properties")
    ax.set(title=f"Q4. Size vs price (Pearson r = {r:.3f})", xlabel="Size (sq ft)", ylabel="Price (lakhs)")
    _save(fig, "q04_size_vs_price")
    return pd.DataFrame({"pearson_r": [r], "p_value": [p], "spearman_rho": [rho]}).round(4), fig


def q05_outliers(df):
    rows = []
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        hi = q3 + config.OUTLIER_IQR_MULTIPLIER * (q3 - q1)
        n = int((df[col] > hi).sum() + (df[col] < q1 - config.OUTLIER_IQR_MULTIPLIER * (q3 - q1)).sum())
        rows.append({"column": col, "upper_fence": round(hi, 1), "max": df[col].max(),
                     "outliers": n, "outlier_pct": round(100 * n / len(df), 2)})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    sns.boxplot(x=df["Price_per_SqFt"], ax=axes[0], color=PALETTE[4])
    axes[0].set(title="Price per sq ft (Rs)", xlabel="")
    sns.boxplot(x=df["Size_in_SqFt"], ax=axes[1], color=PALETTE[1])
    axes[1].set(title="Size (sq ft)", xlabel="")
    fig.suptitle("Q5. Outliers (IQR rule)", y=1.02)
    _save(fig, "q05_outliers")
    return pd.DataFrame(rows).set_index("column"), fig


# --- 6-10: Location -----------------------------------------------------------------
def q06_ppsf_by_state(df):
    table = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).round(0).to_frame("avg_ppsf")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(table.index[::-1], table["avg_ppsf"][::-1], color=PALETTE[0])
    ax.set_xlim(table["avg_ppsf"].min() * 0.97, table["avg_ppsf"].max() * 1.01)
    ax.set(title="Q6. Average price per sq ft by state (axis zoomed)", xlabel="Rs per sq ft")
    _save(fig, "q06_ppsf_by_state")
    return table, fig


def q07_price_by_city(df):
    table = df.groupby("City")["Price_in_Lakhs"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False).round(2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(table.index[::-1], table["mean"][::-1], color=PALETTE[1])
    ax.set_xlim(table["mean"].min() * 0.98, table["mean"].max() * 1.005)
    ax.set(title="Q7. Average property price by city (axis zoomed)", xlabel="Price (lakhs)")
    _save(fig, "q07_price_by_city")
    return table, fig


def q08_median_age_by_locality(df):
    table = df.groupby("Locality")["Age_of_Property"].median().sort_values()
    summary = pd.DataFrame({
        "localities": [table.size],
        "cities_per_locality_avg": [df.groupby("Locality")["City"].nunique().mean()],
        "median_age_min": [table.min()], "median_age_max": [table.max()],
        "overall_median_age": [df["Age_of_Property"].median()],
    }).round(2)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.histplot(table, bins=20, ax=ax, color=PALETTE[2])
    ax.set(title="Q8. Median property age across the 500 localities", xlabel="Median age (years)", ylabel="Localities")
    _save(fig, "q08_median_age_by_locality")
    return summary, table, fig


def q09_bhk_by_city(df):
    share = pd.crosstab(df["City"], df["BHK"], normalize="index").mul(100).round(1)
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(share, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "% of city's listings"}, ax=ax)
    ax.set(title="Q9. BHK mix by city (% of listings)", xlabel="BHK", ylabel="")
    _save(fig, "q09_bhk_by_city")
    return share, fig


def q10_top_locality_trends(df):
    top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5)
    sub = df[df["Locality"].isin(top5.index)]
    # No sale dates exist, so the "trend" axis is construction year.
    trend = sub.groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean().unstack().rolling(5, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    trend.plot(ax=ax, lw=2)
    ax.set(title="Q10. Price by construction year, top 5 most expensive localities (5-yr rolling mean)",
           xlabel="Year built", ylabel="Avg price (lakhs)")
    _save(fig, "q10_top_locality_trends")
    return top5.round(2).to_frame("avg_price_lakhs"), fig


# --- 11-15: Relationships ------------------------------------------------------------
CORR_COLUMNS = [
    "Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "BHK", "Age_of_Property", "Floor_No", "Total_Floors",
    "Nearby_Schools", "Nearby_Hospitals", "Transport_Score", "Amenity_Count", "Infrastructure_Score",
    "Annual_Growth_Rate", "Future_Price_5Y", "Good_Investment",
]


def q11_correlation(df):
    corr = df[CORR_COLUMNS].corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set(title="Q11. Correlation between numeric features and targets")
    _save(fig, "q11_correlation")
    return corr, fig


def _ppsf_by_count(df, col, title, name, color):
    table = df.groupby(col)["Price_per_SqFt"].agg(["mean", "median", "count"]).round(0)
    r, p = stats.pearsonr(df[col], df["Price_per_SqFt"])
    table.attrs.update({"pearson_r": r, "p_value": p})
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(table.index, table["median"], color=color)
    ax.set_ylim(table["median"].min() * 0.95, table["median"].max() * 1.03)
    ax.set(title=f"{title} (r = {r:.3f}, axis zoomed)", xlabel=col.replace("_", " "), ylabel="Median Rs per sq ft")
    _save(fig, name)
    return table, fig


def q12_schools_vs_ppsf(df):
    return _ppsf_by_count(df, "Nearby_Schools", "Q12. Nearby schools vs price per sq ft", "q12_schools_vs_ppsf", PALETTE[2])


def q13_hospitals_vs_ppsf(df):
    return _ppsf_by_count(df, "Nearby_Hospitals", "Q13. Nearby hospitals vs price per sq ft", "q13_hospitals_vs_ppsf", PALETTE[3])


def q14_price_by_furnished(df):
    order = ["Unfurnished", "Semi-furnished", "Furnished"]
    table = df.groupby("Furnished_Status")["Price_in_Lakhs"].agg(["mean", "median", "count"]).loc[order].round(2)
    table.attrs["anova_p"] = stats.f_oneway(*[df.loc[df["Furnished_Status"] == o, "Price_in_Lakhs"] for o in order]).pvalue
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.violinplot(data=df.sample(30000, random_state=0), x="Furnished_Status", y="Price_in_Lakhs", order=order, ax=ax,
                   hue="Furnished_Status", legend=False, inner="quartile")
    ax.set(title="Q14. Price by furnished status", xlabel="", ylabel="Price (lakhs)")
    _save(fig, "q14_price_by_furnished")
    return table, fig


def q15_ppsf_by_facing(df):
    table = df.groupby("Facing")["Price_per_SqFt"].agg(["mean", "median", "count"]).round(0)
    table.attrs["kruskal_p"] = stats.kruskal(*[g["Price_per_SqFt"] for _, g in df.groupby("Facing")]).pvalue
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(table.index, table["median"], color=PALETTE[5])
    ax.set_ylim(table["median"].min() * 0.97, table["median"].max() * 1.02)
    ax.set(title="Q15. Median price per sq ft by facing direction (axis zoomed)", ylabel="Rs per sq ft")
    _save(fig, "q15_ppsf_by_facing")
    return table, fig


# --- 16-20: Investment, amenities, ownership ---------------------------------------
def _count_chart(df, col, title, name):
    table = df[col].value_counts().to_frame("properties")
    table["share_pct"] = (100 * table["properties"] / len(df)).round(2)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(table.index, table["properties"], color=PALETTE[: len(table)])
    for i, v in enumerate(table["properties"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set(title=title, ylabel="Properties")
    _save(fig, name)
    return table, fig


def q16_owner_type(df):
    return _count_chart(df, "Owner_Type", "Q16. Properties by owner type", "q16_owner_type")


def q17_availability(df):
    return _count_chart(df, "Availability_Status", "Q17. Properties by availability status", "q17_availability")


def q18_parking_vs_price(df):
    table = df.groupby("Parking_Space")["Price_in_Lakhs"].agg(["mean", "median", "count"]).round(2)
    t = stats.ttest_ind(df.loc[df["Parking_Space"] == "Yes", "Price_in_Lakhs"],
                        df.loc[df["Parking_Space"] == "No", "Price_in_Lakhs"], equal_var=False)
    table.attrs.update({"t_stat": t.statistic, "p_value": t.pvalue})
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax, hue="Parking_Space", legend=False)
    ax.set(title=f"Q18. Price with and without parking (Welch t-test p = {t.pvalue:.2f})", xlabel="Parking", ylabel="Price (lakhs)")
    _save(fig, "q18_parking_vs_price")
    return table, fig


def q19_amenities_vs_ppsf(df):
    by_count = df.groupby("Amenity_Count")["Price_per_SqFt"].median().round(0)
    by_amenity = pd.Series({a: df.loc[df[f"Has_{a}"] == 1, "Price_per_SqFt"].median() -
                            df.loc[df[f"Has_{a}"] == 0, "Price_per_SqFt"].median() for a in config.AMENITY_LIST}).round(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].bar(by_count.index, by_count.values, color=PALETTE[2])
    axes[0].set_ylim(by_count.min() * 0.95, by_count.max() * 1.03)
    axes[0].set(title="Median Rs/sq ft by number of amenities", xlabel="Amenities")
    axes[1].bar(by_amenity.index, by_amenity.values, color=PALETTE[3])
    axes[1].axhline(0, color="#334155", lw=0.8)
    axes[1].set(title="Median Rs/sq ft: with minus without")
    fig.suptitle("Q19. Amenities vs price per sq ft", y=1.03)
    _save(fig, "q19_amenities_vs_ppsf")
    return pd.DataFrame({"median_ppsf_by_count": by_count}), by_amenity.to_frame("ppsf_difference"), fig


def q20_transport(df):
    order = ["Low", "Medium", "High"]
    table = df.groupby("Public_Transport_Accessibility").agg(
        median_ppsf=("Price_per_SqFt", "median"),
        avg_growth_pct=("Annual_Growth_Rate", "mean"),
        good_investment_pct=("Good_Investment", "mean"),
    ).loc[order]
    table["good_investment_pct"] *= 100
    table = table.round(2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col, title in zip(axes, table.columns, ["Median Rs/sq ft", "Avg annual growth %", "% Good Investment"]):
        ax.bar(order, table[col], color=PALETTE[:3])
        ax.set(title=title)
    axes[0].set_ylim(table["median_ppsf"].min() * 0.97, table["median_ppsf"].max() * 1.02)
    fig.suptitle("Q20. Public transport accessibility: price vs investment potential", y=1.04)
    _save(fig, "q20_transport")
    return table, fig


# --- Approach Step 2 (investment-focused analyses) -------------------------------
def investment_by_city(df):
    table = df.groupby(["City", "City_Tier"]).agg(
        avg_price=("Price_in_Lakhs", "mean"), median_ppsf=("Price_per_SqFt", "median"),
        avg_growth_pct=("Annual_Growth_Rate", "mean"), good_investment_pct=("Good_Investment", "mean"),
    ).reset_index().sort_values("avg_growth_pct", ascending=False)
    table["good_investment_pct"] *= 100
    table = table.round(2)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [PALETTE[2] if t == 2 else PALETTE[0] for t in table["City_Tier"]][::-1]
    ax.barh(table["City"][::-1], table["good_investment_pct"][::-1], color=colors)
    ax.set(title="Share of Good Investments by city (green = Tier 2, blue = Tier 1 metro)", xlabel="% Good Investment")
    _save(fig, "s1_investment_by_city")
    return table, fig


def area_vs_return(df):
    bins = pd.cut(df["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000], labels=["<=1k", "1-2k", "2-3k", "3-4k", "4-5k"])
    table = df.groupby(bins, observed=True).agg(
        avg_growth_pct=("Annual_Growth_Rate", "mean"), median_ppsf=("Price_per_SqFt", "median"),
        good_investment_pct=("Good_Investment", "mean"))
    table["good_investment_pct"] *= 100
    corr = df[["Size_in_SqFt", "Annual_Growth_Rate", "Good_Investment"]].corr().loc["Size_in_SqFt"].round(3)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(table.index.astype(str), table["good_investment_pct"], color=PALETTE[1])
    ax.set(title="Area vs investment return: % Good Investment by size band", xlabel="Size (sq ft)", ylabel="% Good Investment")
    _save(fig, "s2_area_vs_return")
    return table.round(2), corr, fig


def security_vs_investment(df):
    """The brief asks about crime rate, which the dataset does not contain; security is the closest proxy."""
    table = df.groupby("Security").agg(avg_growth_pct=("Annual_Growth_Rate", "mean"),
                                       good_investment_pct=("Good_Investment", "mean"))
    table["good_investment_pct"] *= 100
    return table.round(2)


def infrastructure_vs_resale(df):
    bands = pd.cut(df["Infrastructure_Score"], [0, 30, 45, 60, 75, 101], labels=["<30", "30-45", "45-60", "60-75", "75+"])
    gain = ((df["Future_Price_5Y"] / df["Price_in_Lakhs"] - 1) * 100).rename("gain")
    table = df.assign(gain=gain).groupby(bands, observed=True).agg(
        avg_growth_pct=("Annual_Growth_Rate", "mean"), avg_5y_gain_pct=("gain", "mean"),
        good_investment_pct=("Good_Investment", "mean"), properties=("ID", "count"))
    table["good_investment_pct"] *= 100
    r = stats.pearsonr(df["Infrastructure_Score"], df["Annual_Growth_Rate"])[0]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(table.index.astype(str), table["avg_5y_gain_pct"], color=PALETTE[0])
    ax.set(title=f"Infrastructure score vs 5-year resale gain (r with growth = {r:.2f})",
           xlabel="Infrastructure score (0-100)", ylabel="Avg 5-yr gain %")
    _save(fig, "s3_infrastructure_vs_resale")
    return table.round(2), fig


# --- Statistical significance -------------------------------------------------
def chi_square_tests(df) -> pd.DataFrame:
    """Is each categorical feature independent of Good_Investment? Chi-square test of independence,
    with Cramer's V for effect size (a tiny p-value on 250k rows can still be a negligible effect)."""
    rows = []
    for col in ["Public_Transport_Accessibility", "Availability_Status", "Property_Type", "City_Tier", "Security",
                "Parking_Space", "Furnished_Status", "Facing", "Owner_Type"]:
        table = pd.crosstab(df[col], df["Good_Investment"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
        rows.append({"feature": col, "chi2": round(chi2, 1), "dof": dof, "p_value": p, "cramers_v": round(v, 4),
                     "significant_at_5pct": p < 0.05})
    return pd.DataFrame(rows).set_index("feature").sort_values("cramers_v", ascending=False)


def group_difference_tests(df) -> pd.DataFrame:
    tests = []
    ready = df.loc[df["Availability_Status"] == "Ready to Move", "Annual_Growth_Rate"]
    uc = df.loc[df["Availability_Status"] == "Under Construction", "Annual_Growth_Rate"]
    t = stats.ttest_ind(uc, ready, equal_var=False)
    tests.append({"test": "Welch t-test", "question": "Growth: under construction vs ready to move",
                  "statistic": t.statistic, "p_value": t.pvalue})
    groups = [g["Annual_Growth_Rate"] for _, g in df.groupby("Public_Transport_Accessibility")]
    f = stats.f_oneway(*groups)
    tests.append({"test": "One-way ANOVA", "question": "Growth across transport levels",
                  "statistic": f.statistic, "p_value": f.pvalue})
    k = stats.kruskal(*[g["Price_per_SqFt"] for _, g in df.groupby("City")])
    tests.append({"test": "Kruskal-Wallis", "question": "Price per sq ft across the 42 cities",
                  "statistic": k.statistic, "p_value": k.pvalue})
    r = stats.pearsonr(df["Size_in_SqFt"], df["Price_in_Lakhs"])
    tests.append({"test": "Pearson correlation", "question": "Size vs price",
                  "statistic": r.statistic, "p_value": r.pvalue})
    return pd.DataFrame(tests).set_index("test").round({"statistic": 4})
