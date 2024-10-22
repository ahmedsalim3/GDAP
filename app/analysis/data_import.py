import streamlit as st
from app.functions import fetch_target_data, fetch_ppi_data
from app.ui import display_fetched_data, page_intro


page1_content = """
Here, you can fetch the disease data (targets) from the open target platform using two main ways, whether to fetch using GraphQL, or BigQuery database, etc

... SHORT CONTENT ...

-----
"""

@page_intro(page1_content)  
def run():

    col1, col2, col3 = st.columns([5,5, 2], gap='small', vertical_alignment='top')
    with col1:
        st.header("Disease Data")
        fetch_target_data()
    with col2:
        st.header("String Database")
        fetch_ppi_data()
    with col3:
        if "process_tracker" in st.session_state:
            for stage, status in st.session_state["process_tracker"].items():
                st.write(f"{stage}: {status}")
    
    col1, col2 = st.columns([10, 2])

    with col1:
        display_fetched_data()

        if (
            st.session_state.get("ot_df") is not None
            and st.session_state.get("ppi_df") is not None
        ):
            st.success("Both Target and PPI data fetched successfully!")
        if st.session_state.get("graph") is not None:
            st.success(
                f"{st.session_state['disease_name']} and PPI Bigraph created with {len(st.session_state['positive_edges'])} positive edges and {len(st.session_state['negative_edges'])} negative edges!"
            )


# if __name__ == "__main__":
#     run()

run()