import pandas as pd
import streamlit as st

from tqdm import tqdm
#from modelling import model_fit
from streamlit_option_menu import option_menu
import Forecast,Visualize,Predict
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

def run():
    # app = st.sidebar(
    with st.sidebar:        
        app = option_menu(
            menu_title='Pick your option',
            options=['Home','Forecast','Visualize','Predict','Chat'],
            #icons=['house-fill','person-circle','trophy-fill','chat-fill','info-circle-fill'],
            menu_icon='chat-text-fill',
            default_index=1,
            styles={
                "container": {"padding": "5!important","background-color":'black'},
    "icon": {"color": "white", "font-size": "23px"}, 
    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
    "nav-link-selected": {"background-color": "#02ab21"},})

    forecast_template = pd.read_csv("DemandForecast_template.csv")    
    st.sidebar.download_button(label="Click to download a forecast template",data=forecast_template,file_name='forecast_template.csv',mime='text/csv')
        
    st.sidebar.subheader("Your dataset")
    file = st.sidebar.file_uploader("Upload your document here", type={"csv"})
        
    if app == "Home":
        home(file)
    elif app == "Forecast":
        if "p_df" in st.session_state:
            Forecast.app(st.session_state.p_df)
    elif app == "Visualize":
        #if "model_output" in st.session_state:
        Visualize.app()#st.session_state.model_output
    elif app == "Predict":
        Predict.app(st.session_state.p_df)

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



def home(file):
    if "Process_state" not in st.session_state:
        st.session_state.Process_state = False
    #if "file_state" not in st.session_state:
    #    st.session_state.file_state = None
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

    #st.subheader("Your dataset")
    #file = st.file_uploader("Upload your document here", type={"csv"})
    #st.session_state_file_state = file
    #or st.session_state_file_state is not None

    if st.sidebar.button("Process") or st.session_state.Process_state :
        st.session_state.Process_state = True
        with st.spinner("Processing"):
            # Read the data
            df = pd.read_csv(file)

            # Preprocess the dataframe
            p_df = get_processed_df(df)
            st.session_state.p_df = p_df

            # Get min and max date
            st.write("Select start and end dates to view Preprocessed sample data")
            col1, col2 = st.columns((2))
            startDate = pd.to_datetime(p_df["ActualSaleDate"]).min()
            endDate = pd.to_datetime(p_df["ActualSaleDate"]).max()
            with col1:
                st.session_state.date1 = pd.to_datetime(st.date_input("Start Date", startDate))

            with col2:
                st.session_state.date2 = pd.to_datetime(st.date_input("End Date", endDate))

            if st.session_state.date1 is not None and st.session_state.date2 is not None:
                display_data_button = st.button("Display Data", key="display_data_button")
                # Displaying data
                if display_data_button:
                    with st.spinner("Displaying"):
                        display_data(p_df, st.session_state.date1, st.session_state.date2)
    #return p_df

if __name__ == '__main__':
    #st.sidebar.subheader("Your dataset")
    #file = st.sidebar.file_uploader("Upload your document here", type={"csv"})
    #p_df = home(file)
    run()