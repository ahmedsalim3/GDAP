###################################################################################################################################
# First page of the Analysis section:
#  - It starts by fetching disease-target data from the Open Target platform, using either the GraphQL API or the BigQuery client.
#  - Then loads the STRING database, filtering interactions based on the number of interactions, and displays the dataframes.
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state
from streamlit_helpers.cache_data import (
    fetch_bq_direct_scores,
    fetch_bq_indirect_scores,
    fetch_graphql_data,
    convert_df,
    fetch_ppi_db,
)

from utils import state
from utils import ui as UI

# from utils.style import page_layout


# ==========================
# PAGE LAYOUT AND INTERFACE
# ==========================

# Set the page layout
# page_layout(max_width='100%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')

st.markdown(
    "<h2 style='text-align: center;  color: black;'>Data Collection",
    unsafe_allow_html=True,
)

# st.write(
#     """
#     To make gene predictions associated with a specific disease, you first need to fetch the relevant data.
#     This can be done by obtaining a Disease EFO-ID (Experimental Factor Ontology Identifier) from the
#     [Open Target Platform](https://platform.opentargets.org/). Once you have the EFO-ID, you can retrieve the
#     disease's data (such as targets) using one of two available data sources:

#     1. **GraphQL API**
#     2. **BigQuery Database** (Direct or Indirect Scores)

#     After obtaining the disease-related data, you can enrich your analysis by fetching **Protein-Protein Interaction (PPI) data**
#     from the [STRING database](https://string-db.org/). This data helps identify interactions between proteins that are
#     potentially related to the disease. You can customize the number of PPI interactions you wish to retrieve based
#     on your analysis needs.

#     Choose the appropriate data source for disease data and specify the desired PPI interactions to begin the process.
#     """
# )

st.divider()

# Columns setup
col1, col2, col3 = st.columns([3.5, 5, 2], gap="small", vertical_alignment="top")
# ~parameters ~displaying data ~status

# ==========================
# PARAMETERS SECTION
# ==========================

with col1.container(border=True):
    st.markdown(
        '<h3 style="text-align: center;">Parameters</h3>', unsafe_allow_html=True
    )
    # st.markdown("-----")

    # ================ OPEN-TAGETS ================
    
    st.markdown("##### Disease Data")
    state.init_values("previous_disease_id")

    left, right = st.columns(2, vertical_alignment="top")
    disease_id = left.text_input("Enter Disease EFO-ID:")

    # Example EFO-IDs for quick selection
    example_efo_ids = {
        "Cardiovascular": "EFO_0000319",  # https://platform.opentargets.org/disease/EFO_0000319/associations
        "Alzheimer": "MONDO_0004975",  # https://platform.opentargets.org/disease/MONDO_0004975/associations
    }

    disease_example = right.selectbox(
        "Or select an example disease:",
        options=["Select an example disease"] + list(example_efo_ids.keys()),
    )

    if disease_example != "Select an example disease":
        disease_id = example_efo_ids[disease_example]

    data_source = left.selectbox(
        "Select Data Source:",
        options=["GraphQL", "Direct Scores (BigQuery)", "Indirect Scores (BigQuery)"],
    )

    # Fetch target data button
    right.markdown("<br>", unsafe_allow_html=True)
    ot_button = right.button("Fetch Target Data")
    st.markdown("-----")

    # ================ STRING-DATABASE ================

    st.markdown("##### String Database")
    max_ppi_interactions = st.slider(
        "Maximum number of PPI interactions",
        min_value=50,
        max_value=5000000,
        step=500,
        value=2500000,
        help="This depends solely on the length of the negative genes and the potential positives.",
    )
    ppi_button = st.button("Fetch PPI Data")

# ==========================
# FETCHING DATA
# ==========================

# ================ OPEN-TAGETS DATASET ================

if ot_button:
    # Check if the disease ID has changed
    if _state.previous_disease_id != disease_id:
        # Clears previous data in the session state
        state.delete_state(delete_all=True)
        state.persist("previous_disease_id", disease_id)

    if disease_id:
        state.persist("disease_id", disease_id)

        # Fetch data based on the selected data source
        with col1.status("Fetching disease data...") as fetching:
            params = {"disease_id": disease_id}
            try:
                if data_source == "GraphQL":
                    ot_df, disease_name = fetch_graphql_data(disease_id)
                elif data_source == "Direct Scores (BigQuery)":
                    ot_df, disease_name = fetch_bq_direct_scores(params)
                elif data_source == "Indirect Scores (BigQuery)":
                    ot_df, disease_name = fetch_bq_indirect_scores(params)
                fetching.update(
                    label=f"{disease_name} data Fetched successfully!", state="complete", expanded=False
                )
            except Exception as e:
                col1.error("Please enter a valid EFO-ID.")
                ot_df, disease_name = None, None
                fetching.update(label=f"Failed to fetch disease data", state="error")

        if ot_df is not None:
            state.persist("ot_df", ot_df)
            state.persist("disease_name", disease_name)
            state.persist("data_source", data_source)
            UI.flash_message(
                "ot_df",
                message=f"{_state.disease_name} related data fetched successfully!",
                col=col1,
            )
            UI.task_status(f"Fetching Target dataset", "✅")
    else:
        col1.error("Please enter a valid EFO-ID.")

# ================ STRING-DATABASE ================

if ppi_button:
    with col1.status("Processing protein-protein interactions...") as loading:
        if state.check_state("ot_df"):
            ppi_df = fetch_ppi_db(max_ppi_interactions)
            loading.update(label=f"PPI Data fetched successfully!", state="complete")
        else:
            col1.error("Please fetch disease data first")
            loading.update(label=f"Failed to load PPI data", state="error")
            ppi_df = None

    if ppi_df is not None:
        state.persist("ppi_df", ppi_df)
        UI.task_status("Loading PPI", "✅")
        UI.flash_message("ppi_df", message="PPI Data fetched successfully!", col=col1)
        UI.flash_message(
            "ppi_df",
            "ot_df",
            message="Both Target and PPI data fetched successfully!",
            col=col1,
        )

# ==========================
# DISPLAYING RESULTS
# ==========================

if state.check_state("ot_df", "ppi_df"):
    tab1, tab2 = col2.tabs(["🗃 open-target dataset", "🗃 string database"])

# Display Open Targets Data
if state.check_state("ot_df"):
    expander = tab1.expander(
        f"Click here to see {_state.disease_name} data", expanded=True
    )
    expander.dataframe(_state.ot_df, width=1000)

    c1, c2, c3 = expander.columns([2, 1, 1])
    c1.download_button(
        label=f"Download {_state.disease_name} Dataset",
        data=convert_df(_state.ot_df),
        file_name=f"{_state.disease_name}.csv",
        mime="text/csv",
    )

    # Link to the source
    c2.link_button(
        "Source",
        f"https://platform.opentargets.org/disease/{_state.disease_id}/associations",
    )

# Display PPI Data
if state.check_state("ppi_df"):
    expander2 = tab2.expander("Click here to see PPI data", expanded=True)
    expander2.dataframe(_state.ppi_df, width=1000)

    expander2.download_button(
        label="Download PPI data",
        data=convert_df(_state.ppi_df),
        file_name="PPI_data.csv",
        mime="text/csv",
    )

# Show task status in the right column
UI.show_task_status(col3)

# DELETE IN PRODUCTION
state.manage_state(col3)
