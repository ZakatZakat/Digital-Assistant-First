from src.utils.check_serp_response import check_api_keys
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

st.title("API Keys Status Checker")
st.write("This app allows you to view and update the status of your API keys.")

if st.button('Update API status'):
    df = check_api_keys('api_keys_status.xlsx')
    
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(min_column_width=100)
    gb.configure_grid_options(domLayout='normal')
    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
        height=450,  # можно настроить высоту
        width='100%',
        key="updated_grid"
    )