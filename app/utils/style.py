###################################################################################################################################
#  Utility functions for customizing Streamlit app layout, including footers, sidebar branding, and page styling.
###################################################################################################################################

from pathlib import Path

import streamlit as st

PROJ_ROOT = Path(__file__).resolve().parents[2]

STEMAWAY_LOGO = str(PROJ_ROOT / "docs/icons/stemaway.svg")


def footer() -> None:
    st.markdown(
        """
        <style>
            .image {
                padding: 10px;
                transition: transform .2s;
            }

            .image:hover {
                transform: scale(1.5);
                transition: 0.2s;
            }

            .footer {
                position: relative;
                width: 100%;
                left: 0;
                bottom: 0;
                background-color: white;
                margin-top: auto;
                color: black;
                padding: 0;  /* 5px */
                text-align: center;
                margin: 0;
            }


            .footer a {
                color: #436377;
                text-decoration: none;
                font-family: system-ui, sans-serif;
            }

            .footer a:hover {
                text-decoration: underline;
            }

        .footer p {
                font-size: 13px;  /* Move the inline font-size here */
            }
        </style>

        <div class="footer">
            <p>© 2024 <a href="https://stemaway.com/" target="_blank">stemaway.com</a>. All rights reserved.</p>
            <a href="https://mentorchains.github.io/BI-ML_Disease-Prediction_2024_Site/">
                <img class="image" src="https://d1xykt6w2ydx2s.cloudfront.net/optimized/2X/f/fb6a414e7a33edeea99be0fbc701a057b8351343_2_32x32.png" alt="github" width="50" height="50">
            </a>
            <a href="https://github.com/mentorchains/BI-ML_Disease-Prediction_2024">
                <img class="image" src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="github" width="55" height="55">
            </a>
        </div>
    """,
        unsafe_allow_html=True,
    )


page_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""


def page_layout(
    max_width,
    padding_top="0rem",
    padding_right="0rem",
    padding_left="0rem",
    padding_bottom="0rem",
) -> None:
    st.markdown(
        f"""
        <style>
            .appview-container .main .block-container{{
                max-width: {max_width};
                padding-top: {padding_top};
                padding-right: {padding_right};
                padding-left: {padding_left};
                padding-bottom: {padding_bottom};
            }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# def st_sidebar_footer():
#     st.sidebar.image(
#         image=STEMAWAY_LOGO,
#         width=160,
#         caption="Check out more of our internship and career advancement programs, workshops and mini-projects at stemaway.com",
#     )


def sidebar_footer() -> None:
    st.sidebar.markdown(
        """
        <style>
            .sidebar-footer {
                position: fixed;
                top: 700px;
                text-align: center;
                font-family: system-ui, sans-serif;
                color: #959396;
                font-size: 12px;
                padding-right: 200px;
            }
            .sidebar-footer a {
                color: #436377; !important;
                text-decoration: none;
            }
            .sidebar-footer img {
                width: 160px;
                margin-bottom: 1rem;
            }
        </style>
        <footer class="sidebar-footer">
            <a href="https://mentorchains.github.io/BI-ML_Disease-Prediction_2024_Site/" target="_blank">
                <img src="https://raw.githubusercontent.com/stemaway-repo/stemaway-unified/master/assets/bulb-icon_Large-Icon-branded.svg" alt="StemAway Logo">
            </a>
            <div>
                Check out more of our internship and<br>
                career advancement programs,<br>
                workshops and mini-projects at<br>
                <a href="https://stemaway.com/" target="_blank">stemaway.com</a>.
            </div>
        </footer>
    """,
        unsafe_allow_html=True,
    )
