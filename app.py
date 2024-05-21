import pandas as pd
import streamlit as st
from tqdm import tqdm
#from modelling import model_fit
from streamlit_option_menu import option_menu
import Forecast,modelling #,Predict
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error,accuracy_score,r2_score
from dateutil.relativedelta import relativedelta
#from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Demand Forecasting app",page_icon=":chart_with_upwards_trend:")
st.title(" :chart_with_upwards_trend: Business Demand Forecast")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

def run():
    forecast_template = pd.read_csv("DemandForecast_template.csv")  
    forecast_template_str = forecast_template.to_csv(index=False)  
    st.sidebar.download_button(label="Click to download a forecast template",data=forecast_template_str,file_name='forecast_template.csv',mime='text/csv')

    st.sidebar.subheader("Your dataset")
    file = st.sidebar.file_uploader("Upload your document here", type={"csv"})

    with st.sidebar:        
        app = option_menu(
            menu_title='',
            options=["Preprocess",'Test','Forecast'],#,'Preprocess','Dashboard','Chat'
            #menu_icon='chat-text-fill',
            default_index=0,
            styles={
                "container": {"padding": "!important","background-color":'grey'},
    "icon": {"color": "white", "font-size": "23px"}, 
    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
    "nav-link-selected": {"background-color": "#02ab21"},})

                
    if app == "Preprocess":
        #if st.sidebar.button("Process") or st.session_state.Process_state:
        home(file)
    
    if app == "Test":
        if st.session_state.preprocess_completed:
            modelling.app(st.session_state['p_df_sku'],st.session_state['p_df_store'],st.session_state.Test_state)#,st.session_state.skudata_button_state,st.session_state.storedata_button_state)
            st.session_state.Test_state = True 
        else:
            st.write("Please Preprocess the data to Model and test it")
    elif app == "Forecast":
        if st.session_state.test_completed_sku and st.session_state.test_completed_store:
            Forecast.app(st.session_state.Model_output_sku,st.session_state.Model_output_store) 
            #,st.session_state.skudata_button_state,st.session_state.storedata_button_state
        else:
            st.write("Please Test the data to Forecast")

def process_data_SKU(df):
    df['str_sku_id'] = df['StoreID'] + '-' + df['ProductID']
    product_list = list(df['str_sku_id'].unique())
    grouped_data = df.groupby('str_sku_id')

    with st.spinner("Preprocessing..."):
        # Display progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        Preprocessed_output = []
        for i,Product in enumerate(product_list):
            Product_data = df[df['str_sku_id'] == Product]

            #try:
            Product_data = get_processed_df_SKU(Product_data)
            Preprocessed_output.append(Product_data)

            #except Exception as e:
            #    st.error(f"Error processing product {product}: {str(e)}")
            #    continue

            # Update progress bar
            progress_bar.progress((i + 1) / len(product_list))
            progress_text.text(f"Progress: {i + 1} / {len(product_list)} Products data is preprocessed")

        Preprocessed_output = pd.concat(Preprocessed_output, ignore_index=True)
        #model_output = pd.DataFrame(model_output[0])
        #st.session_state.Model_output = Preprocessed_output
        return Preprocessed_output
    
def process_data_store(df):
    store_list = list(df['StoreID'].unique())
    #grouped_data = df.groupby('str_sku_id')

    with st.spinner("Preprocessing..."):
        # Display progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        Preprocessed_output = []
        for i,store in enumerate(store_list):
            store_data = df[df['StoreID'] == store]

            #try:
            store_data = get_processed_df_store(store_data)
            Preprocessed_output.append(store_data)

            #except Exception as e:
            #    st.error(f"Error processing store {store}: {str(e)}")
            #    continue

            # Update progress bar
            progress_bar.progress((i + 1) / len(store_list))
            progress_text.text(f"Progress: {i + 1} / {len(store_list)} Stores data is preprocessed")

        Preprocessed_output = pd.concat(Preprocessed_output, ignore_index=True)
        #Preprocessed_output = pd.DataFrame(Preprocessed_output[0])
        #st.session_state.Model_output = Preprocessed_output
        #print(model_output.columns)
        return Preprocessed_output

def get_processed_df_SKU(df):
    df["ActualSaleDate"] = pd.to_datetime(df["ActualSaleDate"],format='%d-%m-%Y')# '%Y-%m-%d'
    f_df = prepare_future_data_SKU(df,7)
    df = pd.concat([df,f_df]).reset_index().drop(columns = ['index'])
    df['Day_of_week'] = df['ActualSaleDate'].dt.strftime('%A')
    cal = calendar()
    holidays = cal.holidays(start='2021-01-01', end='2022-12-31')
    df['Holiday'] = df['ActualSaleDate'].isin(holidays)
    df['Holiday'] = df['Holiday'].astype(int)
    df['Season'] = df['ActualSaleDate'].apply(map_to_season)
    df['str_sku_id'] = df['StoreID'].astype(str)+'-'+df['ProductID'].astype(str)
    df_encoded = pd.get_dummies(df, columns=['Holiday', 'Day_of_week','Season'],prefix=['Holiday', 'Day_of_week','Season'], prefix_sep='_')
    columns_to_fill = ['storeCity', 'storeState', 'StoreZip5']
    df_encoded = fill_nans2(df_encoded, columns_to_fill)
    return df_encoded

def prepare_future_data_SKU(p_df,num_of_days):
    unique_dates = p_df['ActualSaleDate'].unique()
    sorted_dates = sorted(unique_dates, reverse=True)
    last_x_dates = sorted_dates[:num_of_days]
    f_df = p_df[['StoreID','ProductID','storeCity','storeState','StoreZip5']][p_df['ActualSaleDate'].isin(last_x_dates)] #,'Inv_Avail'
    f_start_date = p_df['ActualSaleDate'].max() + relativedelta(days=1)
    f_end_date = p_df['ActualSaleDate'].max() + relativedelta(days=num_of_days)
    f_df['ActualSaleDate'] = pd.date_range(f_start_date,f_end_date)
    return f_df

def get_processed_df_store(df):
    df["ActualSaleDate"] = pd.to_datetime(df["ActualSaleDate"],format='%Y-%m-%d')# '%Y-%m-%d'
    f_df = prepare_future_data_store(df,7)
    df = pd.concat([df,f_df]).reset_index().drop(columns = ['index'])
    df['Day_of_week'] = df['ActualSaleDate'].dt.strftime('%A')
    cal = calendar()
    holidays = cal.holidays(start='2021-01-01', end='2022-12-31')
    df['Holiday'] = df['ActualSaleDate'].isin(holidays)
    df['Holiday'] = df['Holiday'].astype(int)
    df['Season'] = df['ActualSaleDate'].apply(map_to_season)
    #df['str_sku_id'] = df['StoreID'].astype(str)+'-'+df['ProductID'].astype(str)
    df_encoded = pd.get_dummies(df, columns=['Holiday', 'Day_of_week','Season'],prefix=['Holiday', 'Day_of_week','Season'], prefix_sep='_')
    columns_to_fill = ['storeCity', 'storeState', 'StoreZip5']
    df_encoded = fill_nans2(df_encoded, columns_to_fill)
    return df_encoded

def prepare_future_data_store(p_df,num_of_days):
    unique_dates = p_df['ActualSaleDate'].unique()
    sorted_dates = sorted(unique_dates, reverse=True)
    last_x_dates = sorted_dates[:num_of_days]
    f_df = p_df[['StoreID','storeCity','storeState','StoreZip5']][p_df['ActualSaleDate'].isin(last_x_dates)]
    f_start_date = p_df['ActualSaleDate'].max() + relativedelta(days=1)
    f_end_date = p_df['ActualSaleDate'].max() + relativedelta(days=num_of_days)
    f_df['ActualSaleDate'] = pd.date_range(f_start_date,f_end_date)
    return f_df

def display_data(p_df, date1, date2):
    df1 = p_df[(p_df["ActualSaleDate"] >= date1) & (p_df["ActualSaleDate"] <= date2)].copy().sort_values(by="ActualSaleDate")
    numRows = df1.shape[0]
    numcols = df1.shape[1]
    df_height = (numRows + 1) #* 35 + 3
    df_width = numcols +1
    st.write("Your Data looks like this:")
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

    if "date1" not in st.session_state:
        st.session_state.date1 = None
    if "date2" not in st.session_state:
        st.session_state.date2 = None
    if "forecast_completed" not in st.session_state:
        st.session_state.forecast_completed = False
    if "test_completed_store" not in st.session_state:
        st.session_state.test_completed_store = False
    if "test_completed_sku" not in st.session_state:
        st.session_state.test_completed_sku = False
    if "Model_output_sku" not in st.session_state:
        st.session_state.Model_output_sku = None
    if "Model_output_store" not in st.session_state:
        st.session_state.Model_output_store = None
    if "skudata_button_state" not in st.session_state:
        st.session_state.skudata_button_state = False
    if "storedata_button_state" not in st.session_state:
        st.session_state.storedata_button_state = False
        st.session_state.button_click_count = 0
    if "Test_state" not in st.session_state:
        st.session_state.Test_state = False        
    if "Process_state" not in st.session_state:
        st.session_state.Process_state = False
    if "preprocess_completed" not in st.session_state:
        st.session_state.preprocess_completed = False

    #if st.button("Process") or st.session_state.Process_state:
    if file is not None:
        if not st.session_state.preprocess_completed:
            with st.spinner("Processing"):
                # Read the data
                df = pd.read_csv(file)
                df["ActualSaleDate"] = pd.to_datetime(df["ActualSaleDate"],format='%Y-%m-%d')# '%Y-%m-%d'
                df[['StoreID','ProductID']] = df[['StoreID','ProductID']].astype('int').astype('string')
                
                # Preprocess the dataframe
                p_df_sku = process_data_SKU(df)
                st.session_state['p_df_sku']=p_df_sku
                cols = ['StoreID','ActualSaleDate','storeCity','storeState','StoreZip5','QtySold']
                df_st = df[cols]
                df_store = pd.DataFrame(df_st.groupby(['StoreID','ActualSaleDate','storeCity','storeState','StoreZip5'])['QtySold'].sum()).reset_index()
                p_df_store = process_data_store(df_store)
                st.session_state['p_df_store']=p_df_store
                st.session_state.preprocess_completed = True
                #st.session_state.p_df = p_df

                # Get min and max date
                st.write("Select start and end dates to view Preprocessed sample data")
                col1, col2 = st.columns((2))
                startDate = pd.to_datetime(p_df_sku["ActualSaleDate"]).min()
                endDate = pd.to_datetime(p_df_sku["ActualSaleDate"]).max()
                with col1:
                    st.session_state.date1 = pd.to_datetime(st.date_input("Start Date", startDate))

                with col2:
                    st.session_state.date2 = pd.to_datetime(st.date_input("End Date", endDate))

                if st.session_state.date1 is not None and st.session_state.date2 is not None:
                    display_data_button = st.button("Display Data", key="display_data_button")
                    # Displaying data
                    if display_data_button:
                        with st.spinner("Displaying"):
                            display_data(p_df_sku, st.session_state.date1, st.session_state.date2)
        else:
            p_df_sku = st.session_state['p_df_sku']
            p_df_store= st.session_state['p_df_store']
            # Get min and max date
            st.write("Select start and end dates to view Preprocessed sample data")
            col1, col2 = st.columns((2))
            startDate = pd.to_datetime(p_df_sku["ActualSaleDate"]).min()
            endDate = pd.to_datetime(p_df_sku["ActualSaleDate"]).max()
            with col1:
                st.session_state.date1 = pd.to_datetime(st.date_input("Start Date", startDate))

            with col2:
                st.session_state.date2 = pd.to_datetime(st.date_input("End Date", endDate))

            if st.session_state.date1 is not None and st.session_state.date2 is not None:
                display_data_button = st.button("Display Data", key="display_data_button")
                # Displaying data
                if display_data_button:
                    with st.spinner("Displaying"):
                        display_data(p_df_sku, st.session_state.date1, st.session_state.date2)
    else:
            st.write("Please upload your file to Preprocess")

if __name__ == '__main__':
    run()