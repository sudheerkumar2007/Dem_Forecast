import pandas as pd
import streamlit as st
import plotly.express as px
from tqdm import tqdm
from modelling import model_fit
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error,accuracy_score,r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Demand Forecasting app",page_icon=":chart_with_upwards_trend:")
#st.beta_set_page_config( layout='wide',page_title="Demand Forecasting app",page_icon=":chart_with_upwards_trend:")
st.title(" :chart_with_upwards_trend: Business Demand Forecast")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

def map_to_season(date):
    if date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    elif date.month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

def fill_nans(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='ffill', inplace=True)
    return df

def fill_nans2(df, columns_to_fill):
    for col in columns_to_fill:
        df[col].fillna(method='bfill', inplace=True)
    return df

def get_processed_df(df):
    df["ActualSaleDate"] = pd.to_datetime(df["ActualSaleDate"])
    df['Day_of_week'] = df['ActualSaleDate'].dt.strftime('%A')
    cal = calendar()
    holidays = cal.holidays(start='2021-01-01', end='2022-12-31')
    df['Holiday'] = df['ActualSaleDate'].isin(holidays)
    df['Holiday'] = df['Holiday'].astype(int)
    df['Season'] = df['ActualSaleDate'].apply(map_to_season)
    df_encoded = pd.get_dummies(df, columns=['Holiday', 'Day_of_week','Season'],prefix=['Holiday', 'Day_of_week','Season'], prefix_sep='_')
    end_date = pd.to_datetime('2022-06-30')
    columns_to_fill = ['VendorId', 'VendorName', 'RetailerID', 'Retailer']
    df_encoded = fill_nans2(df_encoded, columns_to_fill)
    #df_encoded['Sales_7_Days_Lag'] = df_encoded['QtySold'].shift(7)
    #df_encoded['Previousday_EOD_Inv'] = df_encoded['Inv_eod'].shift(7)
    #df_encoded['Previousday_Inv_morn'] = df_encoded['Inv_morn'].shift(7)
    return df_encoded

def display_data(p_df, date1, date2):
        df1 = p_df[(p_df["ActualSaleDate"] >= date1) & (p_df["ActualSaleDate"] <= date2)].copy().sort_values(by="ActualSaleDate")
        numRows = df1.shape[0]
        numcols = df1.shape[1]
        df_height = (numRows + 1) #* 35 + 3
        df_width = numcols +1
        st.write("Your Data looks like this:")
        #st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        return st.dataframe(df1,hide_index=True,use_container_width = True,height = df_height, width = df_width)

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

def draw_linechart(df):
    linechart = pd.DataFrame(df.groupby(df["ActualSaleDate"])[["QtySold","Pred"]].sum()).reset_index()
    fig2 = px.line(linechart, x = "ActualSaleDate", y=["QtySold","Pred"], labels = {"value": "Qty","SKU":"ProductID"},height=500, width = 1000,template="gridon")#hover_data=[linechart["ProductID"],linechart[ "StoreID"]]
    #fig2 = fig2.update_traces(hovertemplate=df["ProductID"])
    #st.plotly_chart(fig2,use_container_width=True)
    return fig2

def predict_data(df):
    data = get_processed_df(df)
    product_data = df[df['str_sku_id'] == product]
    product_data = model_fit(product_data)
    model_output = pd.concat(model_output, ignore_index=True)
    #    st.session_state.Model_output = model_output
    model_output['WMAPE'] = model_output['WMAPE'].replace([np.inf, -np.inf], 100)
    model_output['Model'] = 'RF_with_Season_No_invmorn_previnv_outliers_removed'
    return model_output

def main():
    if "Next_state" not in st.session_state:
        st.session_state.Next_state = False
    if "Display_state" not in st.session_state:
        st.session_state.Display_state = False
    if "vd_state" not in st.session_state:
        st.session_state.vd_state = False
    if "date1" not in st.session_state:
        st.session_state.date1 = None
    if "date2" not in st.session_state:
        st.session_state.date2 = None
    if "forecast_completed" not in st.session_state:
        st.session_state.forecast_completed = False
    if "Model_output" not in st.session_state:
        st.session_state.Model_output = None
    if "visualize" not in st.session_state:
        st.session_state.visualize = False
    if "predict" not in st.session_state:
        st.session_state.predict_state = False

    st.sidebar.subheader("Your dataset")
    file = st.sidebar.file_uploader("upload your document here",type={"csv"})

    if st.sidebar.button("Preprocess") or st.session_state.Next_state:
        st.session_state.Next_state = True
        with st.spinner("processing"):
            #Read the data
            df = pd.read_csv(file)

            #Preprocess the dataframe
            p_df = get_processed_df(df)
            st.write("Preprocessing complete, You can view/Forecast data now")

        if st.sidebar.button("View data") or st.session_state.vd_state:
            st.session_state.vd_state = True
            # Get min and max date 
            st.write("select start and end dates to view sample data")
            col1, col2 = st.columns((2))
            startDate = pd.to_datetime(p_df["ActualSaleDate"]).min()
            endDate = pd.to_datetime(p_df["ActualSaleDate"]).max()
            with col1:
                st.session_state.date1 = pd.to_datetime(st.date_input("Start Date", startDate))

            with col2:
                st.session_state.date2 = pd.to_datetime(st.date_input("End Date", endDate))
                
            if st.session_state.date1 is not None and st.session_state.date2 is not None:
                cl1, cl2, cl3,cl4 = st.columns((4))
                with cl1:
                    display_data_button = st.button("Display Data", key="display_data_button")
                with cl2:
                    forecast_button = st.button("Forecast", key="forecast_button")
                with cl3:
                    visualize_op_button = st.button("Visualize output",key = "visualize_op_button")
                with cl4:
                    predict_button = st.button("Predict",key = "predict_button")

                #Displaying data
                if display_data_button:
                    with st.spinner("Displaying"):
                        display_data(p_df,st.session_state.date1,st.session_state.date2)

                #Forecasting
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

                #Visualizing output
                if visualize_op_button or st.session_state.visualize:
                    if not st.session_state.forecast_completed:
                        st.write("Please Forecast the data to visualize output")

                    else:
                        st.session_state.visualize = True
                        # Filter the data based on selected Store and SKU
                        f_cast = st.session_state.Model_output
                        st.header("Choose your filters: ")
                        # Create filters for Store and SKU
                        #fl1, fl2 = st.columns((2))
                        #with fl1:
                        Store_filter = st.multiselect("Pick your Store", f_cast["StoreID"].unique())
                        #with fl2:
                        if not Store_filter:
                            p_df1 = f_cast.copy()
                            #chart = draw_linechart(p_df1)
                            #st.plotly_chart(chart,use_container_width=True)
                        else:
                            p_df1 =f_cast[f_cast["StoreID"].isin(Store_filter)]
                            #chart = draw_linechart(p_df1)
                            #st.plotly_chart(chart,use_container_width=True)
                        
                        SKU_filter = st.multiselect("Pick your SKU", p_df1["ProductID"].unique())
                        if not SKU_filter:
                            p_df2 = p_df1.copy()
                            chart = draw_linechart(p_df2)
                            st.plotly_chart(chart,use_container_width=True)
                        else:
                            p_df2 =p_df1[p_df1["ProductID"].isin(SKU_filter)]
                            chart = draw_linechart(p_df2)
                            st.plotly_chart(chart,use_container_width=True)

                if predict_button or st.session_state.predict_state:
                    if not st.session_state.forecast_completed:
                        st.write("Please Forecast the data to Predict")
                    else:
                        with st.form(key="form1"):
                            StoreID = st.selectbox("Select your Store", p_df['StoreID'].unique())
                            ProductID = st.selectbox("Select your Store", p_df['ProductID'].unique())
                            ActualSaleDate = st.date_input("Date")
                            filtered_p_df = p_df[(p_df['StoreID']==StoreID) & (p_df["ProductID"]==ProductID)]
                            Sale_lw = st.selectbox("Units sold a week before",filtered_p_df['QtySold'].unique())
                            Inv_eod = st.number_input("Inventory in Presentation")

                            submit_button = st.form_submit_button(label = 'Predict')
                        if submit_button:
                            with st.beta_expander("Results"):
                                pred_df = pd.DataFrame({})

                    #st.session_state.visualize = False
                    #st.session_state.predict_state = True
                    
                    


if __name__ == '__main__':
	main()