#######################################################################################################################################
# Main entry point for the BI ML Disease Prediction app. Defines the page layout, navigation, and links to different app pages
#######################################################################################################################################

from pathlib import Path
import streamlit as st
from utils.style import footer, sidebar_footer

# from utils.style import page_layout

from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


PROJ_ROOT = Path(__file__).resolve().parents[1]
PAGE_ICON = str(PROJ_ROOT / "docs/icons/stemaway-32x32.png")


def run():

    # ================ MAIN PAGE SETUP ================

    st.set_page_config(
        page_title="BI ML Disease Prediction",
        page_icon=PAGE_ICON,
        initial_sidebar_state="expanded",
        layout="wide",
        menu_items={"About": "Developed by Stem-Away, July 2024 Batch"},
    )

    # page_layout(max_width='80%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')
    sidebar_footer()
    st.logo(PAGE_ICON)

    # ================ PAGES ================

    home_page = st.Page(
        page="views/home.py",
        title="Home",
        icon=":material/home:",
        default=True,
    )

    data_fetcher_page = st.Page(
        page="views/analysis/data_collection.py",
        title="Data Collection",
        icon=":material/chevron_right:",
    )

    graph_page = st.Page(
        page="views/analysis/graph_construction.py",
        title="Graph Construction",
        icon=":material/chevron_right:",
    )

    model_page = st.Page(
        page="views/analysis/model_training.py",
        title="Embedding/Model Selection",
        icon=":material/chevron_right:",
    )

    evaluation_page = st.Page(
        page="views/analysis/model_evaluation.py",
        title="Model Evaluation",
        icon=":material/chevron_right:",
    )

    predictions_page = st.Page(
        page="views/analysis/model_prediction.py",
        title="Model Predictions",
        icon=":material/chevron_right:",
    )

    faqs_page = st.Page(
        page="views/faqs.py", title="FAQs", icon=":material/chevron_right:"
    )

    about_page = st.Page(
        page="views/about.py", title="About us", icon=":material/chevron_right:"
    )

    pg = st.navigation(
        {
            "Overview": [home_page],
            "Analysis": [
                data_fetcher_page,
                graph_page,
                model_page,
                evaluation_page,
                predictions_page,
            ],
            "About": [faqs_page, about_page],
        }
    )

    # ================ RUN NAVIGATION ================

    pg.run()
    st.divider()
    footer()


if __name__ == "__main__":
    run()
