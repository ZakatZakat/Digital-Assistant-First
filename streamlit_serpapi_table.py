from src.utils.check_serp_response import check_api_keys
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

st.title("API Keys Status Checker")
st.write("This app allows you to view and update the status of your API keys.")

# Load the initial DataFrame
df = check_api_keys('api_keys_status.xlsx')

# Display the current table
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(min_column_width=100)
gb.configure_grid_options(domLayout='normal')
grid_options = gb.build()

AgGrid(
    df,
    gridOptions=grid_options,
    fit_columns_on_grid_load=True,
    columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
    height=450,
    width='100%',
    key="initial_grid"
)

# Section to add new values to the table
st.subheader("Add New Entry")

# Input fields for new data
new_name = st.text_input("Name")
new_email = st.text_input("Email")
new_api_key = st.text_input("API Key")
new_number = st.number_input("Number", min_value=0)

# Button to add new data
if st.button('Add New Entry'):
    if new_name and new_email and new_api_key:  # Ensure required fields are not empty
        new_row = {
            'Name': new_name,
            'Email': new_email,
            'API Key': new_api_key,
            'Number': new_number
        }
        # Create a new DataFrame for the new row
        new_df = pd.DataFrame([new_row])
        
        # Concatenate the new row with the existing DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save the updated DataFrame back to the Excel file (or any other storage)
        df.to_excel('api_keys_status.xlsx', index=False)
        
        st.success("New entry added successfully!")
        
        # Display the updated table
        AgGrid(
            df,
            gridOptions=grid_options,
            fit_columns_on_grid_load=True,
            columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
            height=450,
            width='100%',
            key="updated_grid"
        )
    else:
        st.error("Please fill in all required fields (Name, Email, API Key).")

# Button to update the status of existing API keys
if st.button('Update API status'):
    df = check_api_keys('api_keys_status.xlsx')
    
    AgGrid(
        df,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
        height=450,
        width='100%',
        key="updated_grid"
    )