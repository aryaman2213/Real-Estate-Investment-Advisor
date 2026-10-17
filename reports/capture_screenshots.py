"""Capture full-page screenshots of the running Streamlit app into reports/screenshots/.

    streamlit run app/streamlit_app.py --server.port 8517        # in another terminal
    python reports/capture_screenshots.py --url http://localhost:8517

Uses Playwright with the locally installed Google Chrome (pip install playwright; no browser download).
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"
WIDTH = 1440


def settle(page, ms=3500):
    """Wait for Streamlit to finish running and Plotly to draw."""
    page.wait_for_selector('[data-testid="stMain"]', timeout=60_000)
    page.wait_for_function('!document.querySelector(\'[data-testid="stStatusWidget"] [data-testid="stSpinner"]\')',
                           timeout=120_000)
    page.wait_for_timeout(ms)


def full_page(page, name):
    """Streamlit scrolls inside its own container, so grow the viewport to the content height first."""
    height = page.evaluate("""() => Math.max(...['[data-testid="stMain"]', '[data-testid="stMainBlockContainer"]']
        .map(s => document.querySelector(s)).filter(Boolean).map(e => e.scrollHeight))""")
    page.set_viewport_size({"width": WIDTH, "height": int(height) + 40})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / f"{name}.png"))
    page.set_viewport_size({"width": WIDTH, "height": 900})
    print("saved", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8517")
    base = parser.parse_args().url.rstrip("/")
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": WIDTH, "height": 900}, device_scale_factor=1)

        page.goto(base)
        settle(page, 6000)
        full_page(page, "01_advisor_good_investment")

        price = page.get_by_role("spinbutton", name="Asking price (lakhs)")
        price.fill("420")
        page.get_by_role("button", name="Analyse this property").click()
        settle(page)
        full_page(page, "02_advisor_not_good")

        page.goto(f"{base}/explorer")
        settle(page, 5000)
        full_page(page, "03_property_explorer")
        page.get_by_role("textbox", name="Search city, locality or state").fill("Nagpur")
        page.keyboard.press("Enter")
        settle(page)
        full_page(page, "04_explorer_search_nagpur")

        page.goto(f"{base}/locations")
        settle(page, 6000)
        full_page(page, "05_location_insights")

        page.goto(f"{base}/market")
        settle(page, 5000)
        full_page(page, "06_market_analysis")

        page.goto(f"{base}/models")
        settle(page, 4000)
        full_page(page, "07_models_classification")
        page.get_by_role("tab", name="Regression: price in 5 years").click()
        settle(page, 2500)
        full_page(page, "08_models_regression")
        page.get_by_role("tab", name="MLflow").click()
        settle(page, 1500)
        full_page(page, "09_models_mlflow")

        page.goto(f"{base}/about")
        settle(page, 2500)
        full_page(page, "10_about")
        browser.close()


if __name__ == "__main__":
    main()
