import streamlit as st
import modelling
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def forecast_data(df):
    product_list = list(df['str_sku_id'].unique())
    grouped_data = df.groupby('str_sku_id')
    end_date = np.max(df['ActualSaleDate'])-relativedelta(months=1)
    # Create a list of DataFrames using dictionary comprehension
    #data_groups = [group_df.copy() for _, group_df in grouped_data]
    with st.spinner("Running model..."):
        # Display progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        model_output = []
        for i,product in enumerate(product_list):
            product_data = df[df['str_sku_id'] == product]

            #try:
            product_data = modelling.model_fit(product_data,end_date)
            model_output.append(product_data)

            #except Exception as e:
            #    st.error(f"Error processing product {product}: {str(e)}")
            #    continue

            # Update progress bar
            progress_bar.progress((i + 1) / len(product_list))
            progress_text.text(f"Progress: {i + 1} / {len(product_list)} products processed")

        #model_output = pd.concat(model_output, ignore_index=True)
        model_output = pd.DataFrame(model_output[0])
        st.session_state.Model_output = model_output
        print(model_output.columns)
        #model_output['MAPE'] = model_output['MAPE'].replace([np.inf, -np.inf], 100)
        #model_output['Model'] = 'RF_with_Season_No_invmorn_previnv_outliers_removed'
        return model_output


def app(df_sku,df_store):
    #forecast_button = st.sidebar.button("Forecast", key="forecast_button")
    #if forecast_button or st.session_state.forecast_state:
        col1, col2 = st.columns((2))
        with col1:
            skuforecast_button = st.button("Store-SKU Level Forecast", key="skuforecast_button")
        with col2:
            storeforecast_button = st.button("Store Level Forecast", key="storeforecast_button")
        if skuforecast_button:
            cols = ['ActualSaleDate','DAYOFWEEK_NM','Season','StoreID','ProductID','storeCity','storeState','StoreZip5','Pred']
            f_df_sku = df_sku[cols][df_sku['Type']=="Forecasted"].rename(columns={"Pred":"Predicted_Sale_Qty"})
            st.markdown("Here is the forecasted sale for next 7 days at sttore-sku level")
            st.dataframe(f_df_sku)#,height =(m_numRows + 1) * 35 + 3,hide_index=True
            #st.session_state.f_op = f_cast
            error = ((np.round(np.mean(df_sku['WAPE']),2)))#.unique()[0]
            st.write(f"Forecast is done at {error}% error rate")

        if storeforecast_button:
            #t_cast_store = st.session_state.Model_output_store
            cols = ['ActualSaleDate','DAYOFWEEK_NM','Season','StoreID','storeCity','storeState','StoreZip5','Pred']
            f_df_store = df_store[cols][df_store['Type']=="Forecasted"].rename(columns={"Pred":"Predicted_Sale_Qty"})
            st.markdown("Here is the forecasted sale for next 7 days at store level")
            st.dataframe(f_df_store)#,height =(m_numRows + 1) * 35 + 3,hide_index=True
            #st.session_state.f_op = f_cast
            error = ((np.round(np.mean(df_store['WAPE']),2)))#.unique()[0]
            st.write(f"Forecast is done at {error}% error rate")