"""Real Estate Investment Advisor (entry point).

Run from the project root:
    streamlit run app/streamlit_app.py

Sets up the theme, the sidebar filters (used by the explorer, location and market pages) and navigation.
Pages live in views.py; data and model access in data.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib

import streamlit as st

import config
import src.features
import src.modeling

# Streamlit only hot-reloads modules inside app/. config.py and src/ live at the project root, so after a
# redeploy (Streamlit Cloud pulls new code without restarting) they could be stale; reload them each run.
importlib.reload(config)
importlib.reload(src.features)
importlib.reload(src.modeling)

import data  # noqa: E402
import ui  # noqa: E402
import views  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon=":material/real_estate_agent:", layout="wide")
ui.install_theme()

FILTER_KEYS = ["f_state", "f_city", "f_type", "f_bhk", "f_price", "f_size", "f_avail", "f_good"]
# Streamlit forgets a widget's value on runs where it is not drawn (the advisor page hides the filters).
# Re-assigning keeps the choice when the user comes back.
# Defaults live in session state (not in the widget call), so the two never conflict.
FILTER_DEFAULTS = {"f_price": (10, 500), "f_size": (500, 5000)}
for key in FILTER_KEYS:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
    elif key in FILTER_DEFAULTS:
        st.session_state[key] = FILTER_DEFAULTS[key]


def sidebar_filters(df) -> data.Filters:
    with st.sidebar:
        st.markdown("## Filters")
        st.markdown('<p class="sidebar-note">Applied to the explorer, location and market pages. '
                    'Leave a filter empty to include everything.</p>', unsafe_allow_html=True)
        states = st.multiselect("State", sorted(df["State"].unique()), key="f_state")
        city_pool = df.loc[df["State"].isin(states), "City"] if states else df["City"]
        cities = st.multiselect("City", sorted(city_pool.unique()), key="f_city")
        types = st.multiselect("Property type", sorted(df["Property_Type"].unique()), key="f_type")
        bhk = st.multiselect("BHK", [1, 2, 3, 4, 5], key="f_bhk")
        price = st.slider("Price (lakhs)", 10, 500, step=10, key="f_price")
        size = st.slider("Size (sq ft)", 500, 5000, step=100, key="f_size")
        avail = st.multiselect("Availability", ["Ready to Move", "Under Construction"], key="f_avail")
        good = st.toggle("Good Investments only", key="f_good")
        st.divider()
        st.markdown('<p class="sidebar-note">Dataset: india_housing_prices.csv (250,000 listings, 42 cities). '
                    'Models tracked with MLflow.</p>', unsafe_allow_html=True)
    return data.Filters(tuple(states), tuple(cities), tuple(types), tuple(bhk), tuple(price), tuple(size), tuple(avail), good)


missing = [p for p in (config.APP_DATA_PATH, config.CLASSIFIER_PATH, config.REGRESSOR_PATH, config.METRICS_PATH) if not p.exists()]
if missing:
    st.error("Data or models not found. From the project root run `python pipeline.py` and then `python train.py`.")
    st.stop()

pages = [
    st.Page(views.advisor, title="Investment advisor", icon=":material/real_estate_agent:", default=True),
    st.Page(views.explorer, title="Property explorer", icon=":material/search:", url_path="explorer"),
    st.Page(views.locations, title="Location insights", icon=":material/map:", url_path="locations"),
    st.Page(views.market, title="Market analysis", icon=":material/query_stats:", url_path="market"),
    st.Page(views.models_page, title="Model performance", icon=":material/model_training:", url_path="models"),
    st.Page(views.about, title="About", icon=":material/info:", url_path="about"),
]
page = st.navigation(pages, position="top")
if page.url_path in ("explorer", "locations", "market"):
    st.session_state["filters"] = sidebar_filters(data.listings())
page.run()
