"""Look and feel: CSS, the Plotly template, and small layout helpers shared by every page."""
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
INK, MUTED, GRID = "#0f172a", "#64748b", "#eef0f4"
ACCENT = "#4f46e5"
GOOD, BAD, WARN = "#059669", "#e11d48", "#d97706"
CHART_COLORS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#e11d48", "#8b5cf6", "#64748b"]
TIER_COLORS = {"Tier 1 metro": "#4f46e5", "Tier 2 city": "#10b981"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, [data-testid="stMetric"], [data-testid="stWidgetLabel"], input, textarea,
[data-baseweb="select"], [data-testid="stSidebar"] p { font-family: %(font)s; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; color: %(ink)s; }

.hero { padding: 0.25rem 0 1.1rem 0; }
.hero .eyebrow { color: %(accent)s; font-weight: 600; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
.hero h1 { font-size: 2.05rem; font-weight: 800; margin: 0.15rem 0 0.35rem 0; padding: 0; }
.hero p { color: %(muted)s; font-size: 0.98rem; margin: 0; max-width: 62rem; }
.pills { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.8rem; }
.pill { background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 0.22rem 0.7rem; font-size: 0.78rem; font-weight: 500; }
.pill.grey { background: #f1f5f9; color: #334155; }

[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e6e8ef; border-radius: 14px;
    padding: 0.9rem 1rem 0.8rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
[data-testid="stMetricLabel"] p { color: %(muted)s; font-size: 0.8rem; font-weight: 500; white-space: normal; }
[data-testid="stMetricLabel"] > div { overflow: visible; }
[data-testid="stMetricValue"] { color: %(ink)s; font-size: clamp(1.05rem, 1.55vw, 1.55rem); font-weight: 700; }
[data-testid="stMetricValue"] > div { white-space: nowrap; overflow: visible; }

[class*="st-key-card-"] { background: #ffffff; border-radius: 14px !important; border-color: #e6e8ef !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); padding: 1rem 1.1rem 0.6rem 1.1rem !important; }
.card-title { font-weight: 650; font-size: 0.98rem; color: %(ink)s; margin: 0 0 0.1rem 0; }
.card-sub { color: %(muted)s; font-size: 0.8rem; margin: 0 0 0.2rem 0; }

.insight { background: #ffffff; border: 1px solid #e6e8ef; border-left: 4px solid %(accent)s; border-radius: 12px;
    padding: 0.85rem 1rem; height: 100%%; }
.insight .num { font-size: 1.35rem; font-weight: 750; color: %(ink)s; }
.insight .txt { color: #334155; font-size: 0.88rem; line-height: 1.45; margin-top: 0.15rem; }

/* verdict banner on the advisor page */
.verdict { border-radius: 16px; padding: 1.1rem 1.3rem; border: 1px solid; display: flex; gap: 1.1rem; align-items: center; }
.verdict.good { background: #ecfdf5; border-color: #a7f3d0; }
.verdict.bad { background: #fff1f2; border-color: #fecdd3; }
.verdict .icon { font-size: 2.3rem; line-height: 1; }
.verdict .label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: %(muted)s; }
.verdict .title { font-size: 1.55rem; font-weight: 800; color: %(ink)s; margin: 0.1rem 0; }
.verdict .sub { color: #334155; font-size: 0.92rem; }
.bar { height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; margin-top: 0.55rem; max-width: 26rem; }
.bar { position: relative; overflow: visible; }
.bar > div { height: 100%%; border-radius: 99px; }
.bar .mark { position: absolute; top: -4px; width: 2px; height: 16px; background: %(ink)s; border-radius: 1px; }
.checks { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.6rem; margin-top: 0.2rem; }
.check { background: #ffffff; border: 1px solid #e6e8ef; border-radius: 12px; padding: 0.7rem 0.9rem; font-size: 0.9rem; color: #334155; }
.check b { color: %(ink)s; }

section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e8ef; }
@media (min-width: 992px) {
  section[data-testid="stSidebar"][aria-expanded="true"] { width: 310px !important; min-width: 310px !important; }
}
@media (max-width: 1180px) {
  [data-testid="stMetric"] { padding: 0.75rem 0.7rem 0.65rem 0.7rem; }
  [data-testid="stMetricValue"] { font-size: clamp(0.98rem, 1.7vw, 1.3rem); }
}
@media (max-width: 640px) {
  .block-container { padding-top: 3.6rem; padding-left: 1rem; padding-right: 1rem; }
  .hero h1 { font-size: 1.6rem; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; }
  .verdict { flex-direction: column; align-items: flex-start; }
}
section[data-testid="stSidebar"] h2 { font-size: 1.05rem; }
.sidebar-note { color: %(muted)s; font-size: 0.78rem; line-height: 1.45; }
</style>
""" % {"font": FONT, "ink": INK, "muted": MUTED, "accent": ACCENT}


# Hover tooltips and drag-to-zoom stay on; the floating toolbar is hidden because it covers legends on phones.
PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}


def install_theme():
    st.markdown(CSS, unsafe_allow_html=True)
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=12.5),
        colorway=CHART_COLORS,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#d7dbe3", ticks=""),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#d7dbe3", ticks=""),
        hoverlabel=dict(bgcolor="white", bordercolor="#d7dbe3", font=dict(family=FONT, color=INK)),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11.5)),
        bargap=0.25,
    )
    pio.templates["reia"] = template
    pio.templates.default = "plotly_white+reia"
    px.defaults.template = "plotly_white+reia"


def hero(eyebrow, title, subtitle, pills=()):
    pill_html = "".join(f'<span class="pill{" grey" if i else ""}">{p}</span>' for i, p in enumerate(pills))
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p>'
        f'{f"<div class=pills>{pill_html}</div>" if pills else ""}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: (label, value, help[, delta[, delta_color]]) tuples rendered as cards."""
    for col, item in zip(st.columns(len(items)), items):
        label, value, help_text, *rest = item
        delta = rest[0] if rest else None
        delta_color = rest[1] if len(rest) > 1 else "normal"
        col.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def card(title):
    return st.container(border=True, key="card-" + slug(title))


def chart_card(fig, title, subtitle=None, height=360, tickangle=0):
    with card(title):
        st.markdown(f'<p class="card-title">{title}</p>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<p class="card-sub">{subtitle}</p>', unsafe_allow_html=True)
        fig.update_layout(template="plotly_white+reia", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=height,
                          margin=dict(l=4, r=8, t=34, b=4), title=None, legend_title_text="",
                          uniformtext_minsize=10, uniformtext_mode="hide")
        fig.update_xaxes(tickangle=tickangle, automargin=True)
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-" + slug(title))


def insight_cards(items):
    """items: (headline number, sentence) pairs."""
    for col, (number, text) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="insight"><div class="num">{number}</div><div class="txt">{text}</div></div>',
                     unsafe_allow_html=True)


def lakhs(value, digits=1):
    """Lakhs -> readable rupees: 85.4 -> Rs 85.4 L, 254 -> Rs 2.54 Cr."""
    if value is None or value != value:
        return "–"
    return f"₹{value / 100:,.2f} Cr" if value >= 100 else f"₹{value:,.{digits}f} L"


def inr(value):
    return "–" if value is None or value != value else f"₹{value:,.0f}"


def probability(p):
    """0.004 -> '<1%', 0.997 -> '>99%', otherwise a whole percentage."""
    return "<1%" if p < 0.005 else ">99%" if p > 0.995 else f"{p:.0%}"


def pct(value, digits=1):
    return "–" if value is None or value != value else f"{value:.{digits}f}%"


FEATURE_LABELS = {
    "PPSF_vs_City_Median": "Price/sq ft vs city median", "Price_per_SqFt": "Price per sq ft",
    "Price_in_Lakhs": "Price", "Log_Price": "Price (log)", "Size_in_SqFt": "Size", "Transport_Score": "Public transport",
    "Infrastructure_Score": "Infrastructure score", "Is_Ready_To_Move": "Ready to move", "City_Tier": "City tier",
    "Property_Type": "Property type", "Amenity_Count": "Number of amenities", "Amenity_Density_Score": "Amenity density",
    "Age_of_Property": "Property age", "Nearby_Schools": "Nearby schools", "Nearby_Hospitals": "Nearby hospitals",
    "Has_Security": "Security", "Has_Parking": "Parking", "Furnished_Level": "Furnishing", "Floor_Ratio": "Floor ratio",
    "Floor_No": "Floor", "Total_Floors": "Total floors", "Owner_Type": "Owner type", "Facing": "Facing", "BHK": "BHK",
    "City": "City", "Has_Clubhouse": "Clubhouse", "Has_Garden": "Garden", "Has_Gym": "Gym",
    "Has_Playground": "Playground", "Has_Pool": "Pool",
}


def label(feature):
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))
