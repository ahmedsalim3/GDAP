#######################################################################################################################################
# Main entry point for the BI ML Disease Prediction app. Defines the page layout, page section using st_pages
#######################################################################################################################################

from pathlib import Path
import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
from utils.style import footer, sidebar_footer

from utils.style import page_layout

from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


PROJ_ROOT = Path(__file__).resolve().parents[1]
PAGE_ICON = str(PROJ_ROOT / "docs/icons/stemaway-32x32.png")
PAGES     = str(PROJ_ROOT / ".streamlit/pages.toml")
SECTIONS  = str(PROJ_ROOT / ".streamlit/pages_sections.toml")


def run():

    # ================ MAIN PAGE SETUP ================

    st.set_page_config(
        page_title='BI ML Disease Prediction',
        page_icon= PAGE_ICON,
        initial_sidebar_state="expanded",
        layout="wide",
        menu_items= {
            'About': 'Developed by Stem-Away, July 2024 Batch'}
    )
    
    # page_layout(max_width='80%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')
    sidebar_footer()
    st.logo(PAGE_ICON)

    sections = st.sidebar.checkbox("Analysis", value=False, key="use_sections")

    nav = get_nav_from_toml(
        SECTIONS if sections else PAGES
    )

    pg = st.navigation(nav)
    add_page_title(pg)
    
    # ================ RUN NAVIGATION ================
    
    pg.run()
    st.divider()
    footer()
    
    
if __name__ == "__main__":
    run()
