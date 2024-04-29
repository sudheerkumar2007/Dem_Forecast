import streamlit as st
from modelling import model_fit
import pandas as pd
import numpy as np
#from app import get_processed_df

def forecast_data(df):
    product_list = list(df['str_sku_id'].unique())
    grouped_data = df.groupby('str_sku_id')
    # Create a list of DataFrames using dictionary comprehension
    data_groups = [group_df.copy() for _, group_df in grouped_data]
    with st.spinner("Running model..."):
        # Display progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        model_output = []
        for i,product in enumerate(product_list):
            product_data = df[df['str_sku_id'] == product]

            try:
                product_data = model_fit(product_data)
                model_output.append(product_data)

            except Exception as e:
                st.error(f"Error processing product {product}: {str(e)}")
                continue

            # Update progress bar
            progress_bar.progress((i + 1) / len(product_list))
            progress_text.text(f"Progress: {i + 1} / {len(product_list)} products processed")

        model_output = pd.concat(model_output, ignore_index=True)
        st.session_state.Model_output = model_output
        model_output['WMAPE'] = model_output['WMAPE'].replace([np.inf, -np.inf], 100)
        model_output['Model'] = 'RF_with_Season_No_invmorn_previnv_outliers_removed'
        return model_output


def app(p_df):
    forecast_button = st.button("Forecast", key="forecast_button")
    if forecast_button :
        if st.session_state.Model_output is None:
        #st.session_state.forecast_state = True
            f_cast = forecast_data(p_df)
            m_numRows = f_cast.shape[0]
            st.write("Forecast is complete. Your output format is here. You can visualize the output now.")
            st.dataframe(f_cast.head(10),hide_index=True)#,height =(m_numRows + 1) * 35 + 3
            st.session_state.forecast_completed = "True"
            st.session_state.Model_output = f_cast
        else:
            st.write("Forecast is already complete. Here is the sample output. It is ready to visualize")
            f_cast = st.session_state.Model_output
            st.dataframe(f_cast.head(10),hide_index=True)