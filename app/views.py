"""The app's pages. Each function renders one page; data access lives in data.py, styling in ui.py."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import data
import ui

TRANSPORT = ["Low", "Medium", "High"]
FURNISHED = ["Unfurnished", "Semi-furnished", "Furnished"]
BENCHMARK = config.GOOD_INVESTMENT_MIN_GROWTH


def _filtered():
    df = data.listings()
    f = st.session_state.get("filters", data.Filters())
    return data.apply_filters(df, f), f


def _empty_guard(df) -> bool:
    if df.empty:
        st.warning("No listings match the current filters. Widen a filter in the sidebar.", icon=":material/filter_alt_off:")
        return True
    return False


# =============================================================================================
# 1. Investment Advisor
# =============================================================================================
DEFAULT_PROPERTY = {
    "State": "Maharashtra", "City": "Nagpur", "Property_Type": "Apartment", "BHK": 3, "Size_in_SqFt": 2400,
    "Price_in_Lakhs": 150.0, "Year_Built": 2018, "Furnished_Status": "Semi-furnished", "Floor_No": 6,
    "Total_Floors": 14, "Nearby_Schools": 7, "Nearby_Hospitals": 6, "Public_Transport_Accessibility": "High",
    "Parking_Space": "Yes", "Security": "Yes", "Amenities": ["Gym", "Garden", "Playground"], "Facing": "East",
    "Owner_Type": "Builder", "Availability_Status": "Under Construction",
}


def advisor():
    m = data.metrics()
    ui.hero(
        "Investment advisor",
        "Is this property a good investment?",
        "Enter a property's details. One model classifies it as a Good Investment or not; a second forecasts its "
        f"price in five years. Both were trained on {m['train_rows']:,} listings, tuned on {m['val_rows']:,} and "
        f"tested on {m['test_rows']:,} they never saw.",
        pills=["Classification: Good Investment", "Regression: price in 5 years", "Confidence + explanation"],
    )
    df = data.listings()
    states = sorted(df["State"].unique())
    d = st.session_state.get("advisor_input", DEFAULT_PROPERTY)

    # State outside the form so the city list can follow it immediately.
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", states, index=states.index(d["State"]), key="adv_state")
    cities = sorted(df.loc[df["State"] == state, "City"].unique())
    city = c2.selectbox("City", cities, index=cities.index(d["City"]) if d["City"] in cities else 0, key="adv_city")

    with st.form("property_form", border=True):
        st.markdown('<p class="card-title">Property details</p>', unsafe_allow_html=True)
        a, b, c, e = st.columns(4)
        ptype = a.selectbox("Property type", ["Apartment", "Independent House", "Villa"],
                            index=["Apartment", "Independent House", "Villa"].index(d["Property_Type"]))
        bhk = b.selectbox("BHK", [1, 2, 3, 4, 5], index=d["BHK"] - 1)
        size = c.number_input("Size (sq ft)", 500, 5000, d["Size_in_SqFt"], step=50,
                              help="500-5,000 sq ft: the range the models were trained on")
        price = e.number_input("Asking price (lakhs)", 10.0, 500.0, float(d["Price_in_Lakhs"]), step=5.0,
                               help="₹10 L to ₹5 Cr (1 crore = 100 lakhs): the range the models were trained on")

        a, b, c, e = st.columns(4)
        year = a.number_input("Year built", 1990, 2023, d["Year_Built"], help="1990-2023: the range in the training data")
        availability = b.selectbox("Availability", ["Ready to Move", "Under Construction"],
                                   index=["Ready to Move", "Under Construction"].index(d["Availability_Status"]))
        floor = c.number_input("Floor number", 0, 30, d["Floor_No"], help="0 = ground floor")
        total_floors = e.number_input("Total floors in building", 1, 30, d["Total_Floors"])

        a, b, c, e = st.columns(4)
        schools = a.slider("Nearby schools", 1, 10, d["Nearby_Schools"])
        hospitals = b.slider("Nearby hospitals", 1, 10, d["Nearby_Hospitals"])
        transport = c.select_slider("Public transport access", TRANSPORT, d["Public_Transport_Accessibility"])
        furnished = e.select_slider("Furnishing", FURNISHED, d["Furnished_Status"])

        a, b, c, e = st.columns(4)
        parking = a.radio("Parking", ["Yes", "No"], index=["Yes", "No"].index(d["Parking_Space"]), horizontal=True)
        security = b.radio("Security (gated / CCTV)", ["Yes", "No"], index=["Yes", "No"].index(d["Security"]), horizontal=True)
        facing = c.selectbox("Facing", ["North", "East", "South", "West"], index=["North", "East", "South", "West"].index(d["Facing"]))
        owner = e.selectbox("Listed by", ["Owner", "Builder", "Broker"], index=["Owner", "Builder", "Broker"].index(d["Owner_Type"]))
        amenities = st.multiselect("Amenities", config.AMENITY_LIST, default=d["Amenities"])
        submitted = st.form_submit_button("Analyse this property", type="primary", icon=":material/insights:")

    form = {
        "State": state, "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": int(size),
        "Price_in_Lakhs": float(price), "Year_Built": int(year), "Furnished_Status": furnished, "Floor_No": int(floor),
        "Total_Floors": int(total_floors), "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
        "Amenities": amenities, "Facing": facing, "Owner_Type": owner, "Availability_Status": availability,
    }
    if submitted:
        if floor > total_floors:
            st.error("Floor number cannot be higher than the building's total floors.", icon=":material/error:")
            return
        if not amenities:
            st.error("Choose at least one amenity (every listing in the training data has one or more).",
                     icon=":material/error:")
            return
        st.session_state["advisor_input"] = form
    analysed = st.session_state.get("advisor_input", DEFAULT_PROPERTY)
    if form != analysed and not submitted:
        st.info("The inputs above have changed. Press **Analyse this property** to update the result below.",
                icon=":material/refresh:")
    _advisor_result(analysed, df)


def _advisor_result(form, df):
    prop = data.build_property(form)
    p = data.predict(prop)
    price = form["Price_in_Lakhs"]
    ppsf_ratio = float(prop["PPSF_vs_City_Median"].iloc[0])
    median_ppsf = data.market_reference()["city_median_ppsf"][form["City"]]

    st.markdown("### Result")
    st.caption(f"For: {form['BHK']} BHK {form['Property_Type'].lower()}, {form['Size_in_SqFt']:,} sq ft in "
               f"{form['City']}, {form['State']} at {ui.lakhs(price)}, built {form['Year_Built']}, "
               f"{form['Availability_Status'].lower()}.")
    # Confidence = probability of the predicted class; strength = how far the score sits from the decision line.
    confidence = p.probability if p.is_good else 1 - p.probability
    margin = abs(p.probability - p.threshold)
    strength = "high" if margin >= 0.25 else "moderate" if margin >= 0.1 else "low (borderline)"
    kind, icon = ("good", "✅") if p.is_good else ("bad", "⚠️")
    title = "Good Investment" if p.is_good else "Not a good investment"
    color = ui.GOOD if p.is_good else ui.BAD
    st.markdown(
        f'<div class="verdict {kind}"><div class="icon">{icon}</div><div>'
        f'<div class="label">Classification model · {strength} confidence</div>'
        f'<div class="title">{title}</div>'
        f'<div class="sub">The model gives this property a <b>{ui.probability(p.probability)}</b> probability of being a Good '
        f'Investment. It says "good" above <b>{p.threshold:.0%}</b> (the cut-off that maximised F1 on validation data). '
        f'Confidence in this verdict: <b>{ui.probability(confidence)}</b>.</div>'
        f'<div class="bar"><div style="width:{p.probability * 100:.1f}%; background:{color}"></div>'
        f'<span class="mark" style="left:{p.threshold * 100:.1f}%"></span></div>'
        f'</div></div>', unsafe_allow_html=True)
    st.write("")

    gain = (p.future_price / price - 1) * 100
    ui.kpi_row([
        ("Asking price", ui.lakhs(price), f"{ui.inr(price * config.LAKH / form['Size_in_SqFt'])} per sq ft"),
        ("Estimated price in 5 years", ui.lakhs(p.future_price), "Regression model forecast", f"{gain:+.1f}% total"),
        ("90% forecast range", f"{ui.lakhs(p.future_low)} – {ui.lakhs(p.future_high)}",
         f"Calibrated on validation data; it contains the true value for "
         f"{100 * data.metrics()['regression']['interval_90_test_coverage']:.1f}% of unseen test listings"),
        ("Implied annual growth", ui.pct(p.implied_growth_pct, 2), f"Benchmark: {BENCHMARK:.0f}% a year",
         f"{p.implied_growth_pct - BENCHMARK:+.2f} pp vs {BENCHMARK:.0f}%"),
    ])
    st.write("")

    growth_ok = p.implied_growth_pct >= BENCHMARK
    price_ok = ppsf_ratio <= config.GOOD_INVESTMENT_MAX_PPSF_RATIO
    st.markdown(
        '<div class="checks">'
        f'<div class="check">{"✅" if price_ok else "❌"} <b>Price check:</b> {ui.inr(price * config.LAKH / form["Size_in_SqFt"])}'
        f' per sq ft is <b>{ppsf_ratio:.0%}</b> of the {form["City"]} median ({ui.inr(median_ppsf)}).</div>'
        f'<div class="check">{"✅" if growth_ok else "❌"} <b>Growth check:</b> forecast growth of '
        f'<b>{p.implied_growth_pct:.2f}%</b> a year is {"above" if growth_ok else "below"} the {BENCHMARK:.0f}% benchmark.</div>'
        '</div>', unsafe_allow_html=True)
    st.caption("A Good Investment must pass both checks. The classifier learned this from data rather than being "
               "told the rule, so its verdict can differ from these two checks when a property is close to either line.")

    left, right = st.columns([1.15, 1])
    with left:
        if p.contributions is not None:
            top = p.contributions.head(8)[::-1]
            fig = go.Figure(go.Bar(
                x=top.values, y=[ui.label(i) for i in top.index], orientation="h",
                marker_color=[ui.GOOD if v > 0 else ui.BAD for v in top.values],
                hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
            fig.update_layout(xaxis_title="← towards 'not good'  ·  towards 'good' →")
            ui.chart_card(fig, "Why the model decided this",
                          "Each bar is how much one feature moved this property's score (SHAP values from XGBoost).", 340)
    with right:
        similar = data.similar_listings(df, prop)
        with ui.card("Similar listings"):
            st.markdown(f'<p class="card-title">Similar listings: {form["BHK"]} BHK {form["Property_Type"].lower()}s in {form["City"]}</p>',
                        unsafe_allow_html=True)
            if similar.empty:
                st.caption("No comparable listings in the dataset.")
            else:
                ui.kpi_row([
                    ("Listings", f"{len(similar):,}", "Same city, type and BHK"),
                    ("Median price", ui.lakhs(similar["Price_in_Lakhs"].median()), "Of the similar listings"),
                    ("Good Investments", ui.pct(100 * similar["Good_Investment"].mean(), 0), "Share of similar listings"),
                ])
                st.write("")
                fig = px.histogram(similar, x="Price_in_Lakhs", nbins=30, color_discrete_sequence=["#cbd5e1"])
                fig.add_vline(x=price, line_color=ui.ACCENT, line_width=3, annotation_text="This property",
                              annotation_position="top")
                fig.update_layout(xaxis_title="Price (lakhs)", yaxis_title="Listings", height=210, margin=dict(l=4, r=8, t=24, b=4),
                                  paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
                st.plotly_chart(fig, width="stretch", theme=None, config=ui.PLOTLY_CONFIG)


# =============================================================================================
# 2. Property Explorer
# =============================================================================================
def explorer():
    df, f = _filtered()
    ui.hero("Property explorer", "Find investment opportunities",
            "Every listing scored by both models. Filter in the sidebar, search, then sort the table by any column.",
            pills=[f.describe()])
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search city, locality or state", placeholder="e.g. Pune, Locality_84", key="explorer_search")
    sort = c2.selectbox("Sort by", ["Model probability (high → low)", "Forecast gain (high → low)",
                                    "Price (low → high)", "Price per sq ft (low → high)"], key="explorer_sort")
    limit = c3.selectbox("Rows to show", [100, 500, 1000, 5000], index=1, key="explorer_rows")
    if query:
        q = query.strip().lower()
        df = df[df["City"].str.lower().str.contains(q, regex=False) | df["Locality"].str.lower().str.contains(q, regex=False)
                | df["State"].str.lower().str.contains(q, regex=False)]
    if _empty_guard(df):
        return
    ui.kpi_row([
        ("Listings", f"{len(df):,}", "Matching the sidebar filters and search"),
        ("Median price", ui.lakhs(df["Price_in_Lakhs"].median()), "Median asking price"),
        ("Median price / sq ft", ui.inr(df["Price_per_SqFt"].median()), "Median rupees per square foot"),
        ("Good Investments", ui.pct(100 * df["Good_Investment"].mean()), "Share labelled Good Investment (rule label)"),
        ("Avg forecast 5-yr gain", ui.pct(df["Predicted_Gain_Pct"].mean()), "Regression model, average over these listings"),
    ])
    st.write("")
    by, asc = {"Model probability (high → low)": ("Model_Probability", False),
               "Forecast gain (high → low)": ("Predicted_Gain_Pct", False),
               "Price (low → high)": ("Price_in_Lakhs", True),
               "Price per sq ft (low → high)": ("Price_per_SqFt", True)}[sort]
    view = df.sort_values(by, ascending=asc).head(limit)
    table = view[["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                  "Availability_Status", "Public_Transport_Accessibility", "Model_Probability", "Predicted_Price_5Y",
                  "Predicted_Gain_Pct", "Model_Verdict", "Good_Investment"]].copy()
    table["Model_Probability"] *= 100
    table["Model_Verdict"] = table["Model_Verdict"].map({1: "✅ Good", 0: "—"})
    table["Good_Investment"] = table["Good_Investment"].map({1: "✅ Good", 0: "—"})
    st.caption(f"Showing {len(view):,} of {len(df):,} matching listings. Click a column header to re-sort. "
               "'Model verdict' is the classifier's prediction; 'Rule label' is the target it was trained to predict.")
    st.dataframe(
        table, hide_index=True, width="stretch", height=520,
        column_config={
            "Property_Type": "Type", "Size_in_SqFt": st.column_config.NumberColumn("Size (sq ft)", format="%,d"),
            "Price_in_Lakhs": st.column_config.NumberColumn("Price (L)", format="%.1f"),
            "Price_per_SqFt": st.column_config.NumberColumn("₹ / sq ft", format="%,.0f"),
            "Availability_Status": "Status", "Public_Transport_Accessibility": "Transport",
            "Model_Probability": st.column_config.ProgressColumn("Good-investment probability", format="%.0f%%",
                                                                 min_value=0, max_value=100),
            "Predicted_Price_5Y": st.column_config.NumberColumn("Forecast in 5 yrs (L)", format="%.1f"),
            "Predicted_Gain_Pct": st.column_config.NumberColumn("Forecast gain", format="%.1f%%"),
            "Model_Verdict": st.column_config.TextColumn("Model verdict", help="Probability above the decision threshold"),
            "Good_Investment": st.column_config.TextColumn("Rule label", help="The target: >= 8% growth and price per sq ft "
                                                                            "at or below the city median"),
        })
    st.download_button("Download these rows (CSV)", view.to_csv(index=False).encode(), "filtered_properties.csv",
                       "text/csv", icon=":material/download:")


# =============================================================================================
# 3. Location insights
# =============================================================================================
def locations():
    df, f = _filtered()
    ui.hero("Location insights", "Where are the good investments?",
            "City and state comparisons of price, forecast growth and the share of Good Investments.",
            pills=[f.describe()])
    if _empty_guard(df):
        return
    city = df.groupby(["City", "State", "Tier_Label"]).agg(
        listings=("ID", "size"), avg_price=("Price_in_Lakhs", "mean"), median_ppsf=("Price_per_SqFt", "median"),
        growth=("Annual_Growth_Rate", "mean"), good=("Good_Investment", "mean")).reset_index()
    city["good"] *= 100
    city["lat"] = city["City"].map(lambda c: config.CITY_COORDINATES[c][0])
    city["lon"] = city["City"].map(lambda c: config.CITY_COORDINATES[c][1])
    best = city.sort_values("good", ascending=False).iloc[0]
    tier = df.groupby("Tier_Label")["Good_Investment"].mean() * 100
    ui.kpi_row([
        ("Cities", f"{city['City'].nunique()}", "Cities in the current selection"),
        ("Best city (share of good)", best["City"], f"{best['good']:.1f}% of its listings are Good Investments"),
        ("Tier 2 cities: Good Investments", ui.pct(tier.get("Tier 2 city", np.nan)), "Developing markets"),
        ("Tier 1 metros: Good Investments", ui.pct(tier.get("Tier 1 metro", np.nan)), "The eight X-class metros"),
    ])
    st.write("")
    metric = st.radio("Colour the map by", ["% Good Investment", "Avg annual growth %", "Median ₹ / sq ft"],
                      horizontal=True, key="loc_metric")
    col = {"% Good Investment": "good", "Avg annual growth %": "growth", "Median ₹ / sq ft": "median_ppsf"}[metric]
    fig = px.scatter_map(city, lat="lat", lon="lon", size="listings", color=col, hover_name="City",
                         hover_data={"State": True, "Tier_Label": True, "listings": ":,", "good": ":.1f",
                                     "growth": ":.2f", "median_ppsf": ":,.0f", "lat": False, "lon": False},
                         color_continuous_scale="Viridis", size_max=26, zoom=3.3, center={"lat": 22.0, "lon": 81.0},
                         map_style="carto-positron", labels={col: metric, "good": "% good", "growth": "growth %",
                                                             "median_ppsf": "median ₹/sq ft", "Tier_Label": "Tier"})
    fig.update_layout(coloraxis_colorbar=dict(title=metric, thickness=12))
    left, right = st.columns([1.1, 1])
    with left:
        ui.chart_card(fig, "Location heatmap", "Bubble size = number of listings; colour = the measure chosen above.", 560)
    with right:
        ranked = city.sort_values(col, ascending=True).tail(15)
        fig = px.bar(ranked, x=col, y="City", color="Tier_Label", orientation="h", color_discrete_map=ui.TIER_COLORS,
                     labels={col: metric, "City": ""})
        lo, hi = ranked[col].min(), ranked[col].max()
        span = (hi - lo) or max(abs(hi) * 0.1, 1)  # a single city has no spread
        fig.update_xaxes(range=[max(lo - span * 0.6, 0), hi + span * 0.1])
        ui.chart_card(fig, "Top 15 cities", f"Ranked by {metric.lower()} (axis zoomed to show differences).", 560)

    heat = df.pivot_table(index="State", columns="Property_Type", values="Good_Investment", aggfunc="mean") * 100
    fig = px.imshow(heat.round(1), text_auto=".1f", color_continuous_scale="Greens", aspect="auto",
                    labels={"color": "% good", "x": "", "y": ""})
    left, right = st.columns(2)
    with left:
        ui.chart_card(fig, "State × property type", "% of listings that are Good Investments.", 600)
    with right:
        choices = sorted(df["City"].unique())
        default = [c for c in ["Mumbai", "Nagpur", "Patna", "Bangalore"] if c in choices][:4] or choices[:3]
        picked = st.multiselect("Cities to compare", choices, default=default, key="trend_cities")
        if not picked:
            st.info("Pick at least one city to draw the trend.", icon=":material/show_chart:")
        else:
            trend = (df[df["City"].isin(picked)].groupby(["Year_Built", "City"])["Price_per_SqFt"].median()
                     .groupby(level="City").transform(lambda s: s.rolling(5, min_periods=1).mean()).reset_index())
            fig = px.line(trend, x="Year_Built", y="Price_per_SqFt", color="City",
                          labels={"Year_Built": "Year built", "Price_per_SqFt": "Median ₹ / sq ft"})
            ui.chart_card(fig, "Price trend by construction year",
                          "No sale dates exist, so year built is the time axis (5-year rolling median).", 520)


# =============================================================================================
# 4. Market analysis (EDA highlights)
# =============================================================================================
def market():
    df, f = _filtered()
    ui.hero("Market analysis", "What the data says",
            "Highlights from the exploratory analysis (all 20 questions are answered in notebooks/eda.ipynb).",
            pills=[f.describe()])
    if _empty_guard(df):
        return
    full = data.listings()
    r_size_price = full["Size_in_SqFt"].corr(full["Price_in_Lakhs"])
    transport = full.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(TRANSPORT) * 100
    ui.insight_cards([
        (f"r = {r_size_price:.3f}", "Size and price are unrelated in this dataset: listings look randomly generated, "
                                    "so no single raw column predicts price."),
        (f"{transport['High']:.0f}% vs {transport['Low']:.0f}%", "of high-transport vs low-transport listings are Good "
                                                                  "Investments: connectivity is the strongest driver."),
        ("Larger = cheaper per sq ft", "Price per sq ft falls as size rises, so big homes pass the 'not overpriced' "
                                       "check far more often."),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Price_in_Lakhs", nbins=50, color_discrete_sequence=[ui.ACCENT],
                           labels={"Price_in_Lakhs": "Price (lakhs)"})
        fig.update_layout(yaxis_title="Listings")
        ui.chart_card(fig, "Price distribution", "Flat from ₹10 L to ₹5 Cr: a uniform, synthetic distribution.")
    with c2:
        sample = df.sample(min(len(df), 40000), random_state=0)
        fig = px.box(sample, x="Property_Type", y="Price_per_SqFt", color="Property_Type", points=False,
                     labels={"Price_per_SqFt": "₹ per sq ft", "Property_Type": ""})
        fig.update_layout(showlegend=False)
        fig.update_yaxes(type="log")
        ui.chart_card(fig, "Price per sq ft by property type", "Log scale. Medians are almost identical across types.")
    c1, c2 = st.columns(2)
    with c1:
        g = df.groupby(["Public_Transport_Accessibility", "Availability_Status"])["Good_Investment"].mean().mul(100).reset_index()
        fig = px.bar(g, x="Public_Transport_Accessibility", y="Good_Investment", color="Availability_Status", barmode="group",
                     category_orders={"Public_Transport_Accessibility": TRANSPORT},
                     labels={"Good_Investment": "% Good Investment", "Public_Transport_Accessibility": "Public transport"})
        ui.chart_card(fig, "Transport and availability vs Good Investment",
                      "Better connectivity and under-construction purchases both raise expected appreciation.")
    with c2:
        bands = pd.cut(df["Infrastructure_Score"], [0, 30, 45, 60, 75, 101], labels=["<30", "30-45", "45-60", "60-75", "75+"])
        g = df.assign(band=bands, gain=(df["Future_Price_5Y"] / df["Price_in_Lakhs"] - 1) * 100).groupby(
            "band", observed=True)["gain"].mean().reset_index()
        fig = px.bar(g, x="band", y="gain", color_discrete_sequence=[ui.GOOD],
                     labels={"band": "Infrastructure score (schools, hospitals, transport)", "gain": "Avg 5-year gain %"})
        ui.chart_card(fig, "Infrastructure vs resale value", "Higher infrastructure score → higher 5-year appreciation.")
    c1, c2 = st.columns(2)
    with c1:
        bands = pd.cut(df["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000], labels=["≤1k", "1-2k", "2-3k", "3-4k", "4-5k"])
        g = df.assign(band=bands).groupby("band", observed=True)["Good_Investment"].mean().mul(100).reset_index()
        fig = px.bar(g, x="band", y="Good_Investment", color_discrete_sequence=["#0ea5e9"],
                     labels={"band": "Size (sq ft)", "Good_Investment": "% Good Investment"})
        ui.chart_card(fig, "Area vs investment outcome", "Same price spread for every size, so bigger homes are cheaper per sq ft.")
    with c2:
        cols = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
                "Infrastructure_Score", "Amenity_Count", "Annual_Growth_Rate", "Good_Investment"]
        corr = df[cols].corr().round(2)
        names = [ui.label(c) for c in cols]
        names[-2], names[-1] = "Annual growth", "Good Investment"
        fig = px.imshow(corr.values, x=names, y=names, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_traces(textfont_size=9)
        ui.chart_card(fig, "Correlation matrix", "Raw columns barely correlate with each other; the targets do with the drivers.",
                      480, tickangle=-40)


# =============================================================================================
# 5. Model performance
# =============================================================================================
def _metric_table(models: dict, champion: str, keys: list[str], labels: list[str]) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        row = {"Model": ("🏆 " if name == champion else "") + name}
        row.update({lab: m.get(k, np.nan) if m.get(k) is not None else np.nan for k, lab in zip(keys, labels)})
        rows.append(row)
    return pd.DataFrame(rows)


def models_page():
    m, ev = data.metrics(), data.evaluation()
    clf, reg = m["classification"], m["regression"]
    ui.hero("Model performance", "How well do the models work?",
            f"{len(clf['models']) - 1} classifiers and {len(reg['models']) - 1} regressors were trained on the same "
            f"{m['train_rows']:,} listings. The champion of each task was picked on {m['val_rows']:,} validation listings, "
            f"then scored once on {m['test_rows']:,} test listings (all figures below are test results). Everything is "
            "tracked in MLflow, and the champions are registered under the alias 'champion'.",
            pills=[f"Classifier: {clf['champion']} (v{clf['registry_version']})",
                   f"Regressor: {reg['champion']} (v{reg['registry_version']})", "Tracked with MLflow"])
    tab_c, tab_r, tab_m = st.tabs(["Classification: Good Investment", "Regression: price in 5 years", "MLflow"])

    with tab_c:
        champ = clf["models"][clf["champion"]]
        base = clf["models"]["Baseline (majority class)"]
        ui.kpi_row([
            ("ROC AUC", f"{champ['roc_auc']:.3f}", "Ranking quality across all thresholds (0.5 = coin flip)"),
            ("F1 score", f"{champ['f1']:.3f}", "Balance of precision and recall for the Good class"),
            ("Accuracy", ui.pct(100 * champ["accuracy"]), "Share of test listings classified correctly",
             f"{100 * (champ['accuracy'] - base['accuracy']):+.1f} pp vs baseline"),
            ("Precision", ui.pct(100 * champ["precision"]), "Of the listings it calls good, how many are"),
            ("Recall", ui.pct(100 * champ["recall"]), "Of the truly good listings, how many it finds"),
        ])
        st.write("")
        table = _metric_table(clf["models"], clf["champion"],
                              ["val_roc_auc", "roc_auc", "f1", "accuracy", "precision", "recall", "threshold", "train_seconds"],
                              ["Val ROC AUC", "ROC AUC", "F1", "Accuracy", "Precision", "Recall", "Threshold", "Train (s)"])
        st.dataframe(table.sort_values("Val ROC AUC", ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in ["Val ROC AUC", "ROC AUC", "F1", "Accuracy", "Precision", "Recall"]}
                     | {"Threshold": st.column_config.NumberColumn(format="%.2f"),
                        "Train (s)": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"Classes are {100 * (1 - clf['positive_rate']):.0f}:{100 * clf['positive_rate']:.0f}, so accuracy alone "
                   "misleads: the baseline that never predicts 'good' already scores "
                   f"{100 * base['accuracy']:.0f}%. The champion is the model with the best validation ROC AUC (F1 breaks ties). "
                   "Class weights (or XGBoost's scale_pos_weight) handle the imbalance, and each model's decision threshold "
                   f"was tuned on validation to maximise F1 (champion: {clf['threshold']:.2f}). Test columns are unprefixed.")
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            cm = np.array(ev["classification"]["confusion_matrices"][clf["champion"]])
            fig = px.imshow(cm, text_auto=",", x=["Not good", "Good"], y=["Not good", "Good"], color_continuous_scale="Blues",
                            labels={"x": "Predicted", "y": "Actual", "color": "Listings"})
            fig.update_layout(coloraxis_showscale=False)
            ui.chart_card(fig, "Confusion matrix", f"{clf['champion']}, {m['test_rows']:,} test listings.", 330)
        with c2:
            roc = ev["classification"]["roc_curve"]
            fig = go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=ui.ACCENT, width=3),
                                        name=f"AUC {champ['roc_auc']:.3f}"),
                             go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94a3b8", dash="dash"), name="Random")])
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
            ui.chart_card(fig, "ROC curve", "Further into the top-left corner is better.", 330)
        with c3:
            _importance_chart(ev["classification"]["feature_importance"].get(clf["champion"]), "Feature importance (classifier)")

    with tab_r:
        champ = reg["models"][reg["champion"]]
        base = reg["models"]["Baseline (fixed 8% rate)"]
        ui.kpi_row([
            ("RMSE", f"{champ['rmse']:.2f} L", "Root mean squared error, in lakhs",
             f"{100 * (champ['rmse'] / base['rmse'] - 1):+.0f}% vs fixed-8% formula", "inverse"),
            ("MAE", f"{champ['mae']:.2f} L", "Mean absolute error, in lakhs"),
            ("R²", f"{champ['r2']:.4f}", "Share of variance in future price explained"),
            ("MAPE", ui.pct(champ["mape_pct"], 2), "Average error as a % of the true future price"),
        ])
        st.write("")
        table = _metric_table(reg["models"], reg["champion"], ["val_rmse", "rmse", "mae", "mse", "r2", "mape_pct", "train_seconds"],
                              ["Val RMSE", "RMSE", "MAE", "MSE", "R²", "MAPE %", "Train (s)"])
        st.dataframe(table.sort_values("Val RMSE"), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Val RMSE", "RMSE", "MAE", "MSE", "MAPE %"]}
                     | {"R²": st.column_config.NumberColumn(format="%.5f"), "Train (s)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Future price = price × (1 + growth)^5 is multiplicative, so every model learns log(price): in log space the "
                   "formula becomes additive, which is why plain linear regression is as good as the boosted trees. "
                   "R² is high for all models because today's price explains most of the future price; RMSE and the "
                   "fixed-rate baseline show the real gain. The champion is the lowest validation RMSE; its 90% forecast "
                   f"range covers {100 * reg['interval_90_test_coverage']:.1f}% of test listings.")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            s = ev["regression"]["test_sample"]
            fig = px.scatter(x=s["actual"], y=s["predicted"], opacity=0.35, color_discrete_sequence=[ui.ACCENT],
                             labels={"x": "Actual price in 5 years (L)", "y": "Predicted (L)"})
            lim = [min(s["actual"]), max(s["actual"])]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=ui.BAD, dash="dash"), name="Perfect"))
            fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
            ui.chart_card(fig, "Actual vs predicted", "3,000 random test listings.", 380)
        with c2:
            _importance_chart(ev["regression"]["feature_importance"].get(reg["champion"]), "Feature importance (regressor)", 380,
                              "|Coefficient| on standardised inputs. Infrastructure score is built from schools, hospitals and transport, "
                              "so those four share one effect between them (multicollinearity)." if "Linear" in reg["champion"] or "Ridge" in reg["champion"] else None)

    with tab_m:
        st.markdown(f"""
**Experiments.** `{config.EXPERIMENT_CLASSIFICATION}` and `{config.EXPERIMENT_REGRESSION}`. Each run logs the model's
hyperparameters, every metric, a confusion matrix or actual-vs-predicted plot, a feature-importance chart and the
fitted pipeline (preprocessing + model) with its input signature.

**Model registry.**

| Registered model | Version | Alias | Model |
|---|---|---|---|
| `{config.REGISTERED_CLASSIFIER}` | {clf['registry_version']} | `{config.CHAMPION_ALIAS}` | {clf['champion']} |
| `{config.REGISTERED_REGRESSOR}` | {reg['registry_version']} | `{config.CHAMPION_ALIAS}` | {reg['champion']} |

**Hyperparameter tuning.** XGBoost was tuned for each task with a {config.TUNING_ITERATIONS}-candidate randomised search,
{config.CV_FOLDS}-fold cross-validated on a {config.TUNING_SAMPLE_SIZE:,}-row sample, and logged as the "XGBoost (tuned)" run.

**Split.** {m['train_rows']:,} train / {m['val_rows']:,} validation / {m['test_rows']:,} test, stratified on Good_Investment.
Validation picks each classifier's decision threshold, the champions and the forecast range; test is scored once.

To browse the runs locally, from the project folder (the script also fixes MLflow's stored paths after a fresh clone):
```bash
python mlflow_ui.py
```
To reload both champions from the registry and re-check every reported test metric:
```bash
python evaluate.py
```
The app loads the champion models exported from the registry (`models/*.joblib`), so it runs without an MLflow server.
""")
        st.json({"classifier_best_params": clf["tuning_best_params"], "regressor_best_params": reg["tuning_best_params"]},
                expanded=False)


def _importance_chart(importance: dict | None, title: str, height=330, subtitle=None):
    if not importance:
        st.info("This model type does not expose feature importances.")
        return
    s = pd.Series(importance).head(12)[::-1]
    fig = go.Figure(go.Bar(x=s.values * 100, y=[ui.label(i) for i in s.index], orientation="h", marker_color=ui.ACCENT,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Share of importance (%)")
    ui.chart_card(fig, title, subtitle or "Top 12 features; one-hot columns summed back to their source feature.", height)


# =============================================================================================
# 6. About
# =============================================================================================
def about():
    ui.hero("About", "Methodology and assumptions",
            "How the dataset was cleaned, how the two targets were defined, and how the models were built.")
    adj = config.GROWTH_ADJUSTMENTS
    m = data.metrics()
    clf, reg = m["classification"], m["regression"]
    st.markdown(f"""
### Problem
Help investors decide whether a property is worth buying (**classification: Good Investment**) and what it will be worth
in five years (**regression: Future_Price_5Y**), using 250,000 Indian property listings.

### Data cleaning
- No missing values or duplicate listings were found. The checks and imputation (median for numbers, mode for
  categories) still run, and the same imputers sit inside the model pipeline to protect the app against blank input.
- **Floor number above total floors in 46.5% of rows**: an impossible combination, almost certainly two swapped
  columns, so the two values were swapped back instead of dropping half the data.
- **Price_per_SqFt** in the raw file is in *lakhs* and rounded to two decimals (0.10 covers ₹9,500–₹10,499). It was
  recomputed exactly, in rupees.
- **Outliers**: 19,723 IQR outliers in price per sq ft, all real (small, expensive homes). Flagged, not removed.
- **Locality** is dropped from the models: each of its 500 values appears in all 42 cities, so it carries no location
  information. **State** is dropped because City determines it.

### Target 1: Future price in 5 years
The dataset has no sale history, so growth follows the brief's *feature-based* method:

`Future_Price_5Y = Price × (1 + r)^{config.FORECAST_YEARS}`

`r` = {config.GROWTH_BASE_RATE}% base + adjustments, plus a random market shock (σ = {config.GROWTH_NOISE_STD} pp) for everything
a listing cannot capture (rates, policy, local news). Bounded to {config.GROWTH_RATE_BOUNDS[0]:.0f}–{config.GROWTH_RATE_BOUNDS[1]:.0f}%.

| Driver | Adjustment (percentage points per year) | Reason |
|---|---|---|
| Public transport | Low {adj['transport']['Low']:+}, Medium {adj['transport']['Medium']:+}, High {adj['transport']['High']:+} | Connectivity drives resale demand |
| Availability | Under construction {adj['availability']['Under Construction']:+} | Pre-launch pricing; value rises on completion |
| Property type | Villa {adj['property_type']['Villa']:+}, Independent house {adj['property_type']['Independent House']:+} | More land per rupee; land appreciates, buildings age |
| City tier | Tier 2 {adj['city_tier'][2]:+} | Developing cities grow faster from a lower base |
| Security / parking | {adj['security']['Yes']:+} each | Gated, parked homes resell at a premium |
| Amenities | {config.GROWTH_PER_AMENITY:+} per amenity | Clubhouse, gym, pool, garden, playground |
| Schools + hospitals | {config.GROWTH_PER_SOCIAL_FACILITY:+} per facility | Social infrastructure nearby |
| Age | {config.GROWTH_PER_YEAR_OF_AGE:+} per year of age above {config.GROWTH_AGE_PIVOT_YEARS} (newer homes gain the same per year below it) | Older buildings depreciate |

The base was chosen so the market-wide average is ≈8%, the rate in the brief's own example.

### Target 2: Good Investment
A property is a **Good Investment** when **both** are true:
1. its annual appreciation is at least **{config.GOOD_INVESTMENT_MIN_GROWTH:.0f}%** (the brief's "appreciation rate > threshold"), and
2. its price per sq ft is **at or below its city's median** (the brief's "price per sq ft cheaper than median").

That makes a good investment one that grows faster than the market **and** is not overpriced to begin with; 27% of
listings qualify.

### Models
- One stratified split, shared by both tasks: {m['train_rows']:,} train / {m['val_rows']:,} validation / {m['test_rows']:,} test.
  Models are fitted on train; validation picks the champions, each classifier's decision threshold (maximising F1) and
  the 90% forecast range; test is scored once and is what every page reports.
- Numeric features scaled, categories one-hot encoded, all inside a scikit-learn pipeline so the app applies exactly
  the same preprocessing. Class imbalance (73:27) handled with class weights / `scale_pos_weight`.
- Classification: baseline, Logistic Regression, Decision Tree, Random Forest, Extra Trees, Hist Gradient Boosting,
  XGBoost, tuned XGBoost. Regression: fixed-8% baseline, Linear, Ridge, Decision Tree, Random Forest, Hist Gradient
  Boosting, XGBoost, tuned XGBoost (all on a log-transformed target).
- Everything logged to MLflow; the champions ({clf['champion']} and {reg['champion']}) are registered with the alias
  `champion`. `python evaluate.py` reloads them from the registry and re-checks every reported test metric.

### Honest limits
The listings are synthetic (every column is independent of every other), and the targets are rule-based. The models
therefore show how well each algorithm recovers a realistic, documented investment rule, not real market behaviour.
With real transaction data the same pipeline would learn the growth drivers instead of having them specified.
""")
