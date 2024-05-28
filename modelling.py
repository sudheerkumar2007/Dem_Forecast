import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import plotly.express as px
from dateutil.relativedelta import relativedelta


if "Test_state" not in st.session_state:
    st.session_state.Test_state = False
if "skudata_button_state" not in st.session_state:
    st.session_state.skudata_button_state = False
if "storedata_button_state" not in st.session_state:
    st.session_state.storedata_button_state = False


def mape(actual, pred): 
    '''
    Mean Absolute Percentage Error (MAPE) Function
    
    input: list/series for actual values and predicted values
    output: mape value 
    '''
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
    
def post_prep_data(chk):
    one_hot_columns = [col for col in chk.columns if col.startswith(('Day_of_week_', 'Holiday','Season_'))]
    chk['DAYOFWEEK_NM'] = (chk[[col for col in chk if col.startswith(('Day_of_week_'))]]==1).idxmax(1)
    chk['Holiday'] = (chk[[col for col in chk if col.startswith(('Holiday_'))]]==1).idxmax(1)
    chk['Season'] = (chk[[col for col in chk if col.startswith(('Season_'))]]==1).idxmax(1)
    chk['DAYOFWEEK_NM'] = chk['DAYOFWEEK_NM'].str.replace('Day_of_week_','')
    chk['Holiday'] = chk['Holiday'].str.replace('Holiday_','')
    chk['Season'] = chk['Season'].str.replace('Season_','')
    lag_columns = [col for col in chk.columns if 'Lag' in col]
    chk = chk.drop(columns=one_hot_columns+lag_columns+['QS_modified'])
    #chk['zero_sales_streak'] = (chk['QtySold'] == 0).astype(int)
    # Use the cumsum() function to assign a unique group number to consecutive zero sales streaks
    #chk['zero_sales_group'] = chk['zero_sales_streak'].cumsum()
    # Use the cumsum() function to assign a unique group number to consecutive zero sales streaks
    #chk['zero_sales_group'] = (chk['QtySold'] != 0).astype(int).cumsum()
    # Calculate the count of consecutive zero sales days for each group
    #chk['consecutive_zero_sales'] = chk.groupby('zero_sales_group')['zero_sales_streak'].cumsum()
    # Filter out the rows where sales are not zero
    #chk['consecutive_zero_sales'] = chk['consecutive_zero_sales'] * (chk['QtySold'] == 0)
    # Drop the intermediate columns if needed
    #chk = chk.drop(['zero_sales_streak', 'zero_sales_group'], axis=1)
    return chk


def model_fit_SKU(chk,end_date,forecast_date):
    #columns_to_fill = ['VendorID', 'VendorName', 'RetailerID', 'RetailerName']
    #print(np.unique(chk['str_sku_id']))
    #chk['Sales_7_Days_Lag'] = chk['proportion_sale'].shift(7)
    chk['Sales_7_Days_Lag'] = chk['QtySold'].shift(14)
    #chk['Inv_Avail_7_Days_Lag'] = chk['Inv_Avail'].shift(7)
    cols_selected = ['QtySold', 'Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday','Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday','Day_of_week_Wednesday',  'Holiday_0', 'Holiday_1','Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter','Sales_7_Days_Lag']#,'Inv_Avail_7_Days_Lag','Inv_Avail','Previousday_EOD_Inv','Inv_morn','tavg', 'wspd',
    #,'Inv_EOD_7_Days_Lag','Morn_Inv_7_Days_Lag'
    cols_to_drop = list(chk.columns[chk.isna().all()]) # finding columns that have all Nan values
    cols_selected = [col for col in cols_selected if col not in cols_to_drop] # removing columns that have all Nan values
    Q1 = chk['QtySold'].quantile(0.25)
    Q3 = chk['QtySold'].quantile(0.75)
    IQR = Q3-Q1
    avg_val = np.mean(chk['QtySold'][(chk['QtySold'] >= Q1-1.5*IQR) & (chk['QtySold'] <= Q3+1.5*IQR)])
    chk['QS_modified'] = np.where(((chk['QtySold'] <= Q1-1.5*IQR) | (chk['QtySold'] >= Q3+1.5*IQR)),avg_val,chk['QtySold'])
    cols_selected.append("QS_modified")
    chk['Type'] = "Train"
    chk_train = chk[cols_selected][chk['ActualSaleDate']<=end_date]
    chk_test = chk[cols_selected][(chk['ActualSaleDate']>end_date) & (chk['ActualSaleDate']<forecast_date)]
    chk_forecast = chk[cols_selected][chk['ActualSaleDate']>=forecast_date]

    #Removing outliers and nans from train data
    chk_train.dropna(inplace=True)
    #chk_train = chk_train[(chk_train['QtySold'] >= Q1-1.5*IQR) & (chk_train['QtySold'] <= Q3+1.5*IQR)]
    label_train = chk_train.pop("QS_modified")
    label_test = chk_test.pop("QS_modified")
    label_forecast = chk_forecast.pop("QS_modified")
    chk_train.pop("QtySold"),chk_test.pop("QtySold"),chk_forecast.pop("QtySold")
    
    if(chk_train.shape[0]>chk_test.shape[0]) :
        #Model fitting
        np.random.seed(475)
        rf = RandomForestRegressor()
        mdl = rf.fit(chk_train,label_train)

        #Prediction on test set
        pred = mdl.predict(chk_test)
        pred = pd.DataFrame(np.round(pred)).rename(columns={0:'Predicted'})

        #Formatting the output and calculating performance metrics
        pred.index = chk_test.index
        chk.loc[chk.index.isin(pred.index),['Pred']] = pred['Predicted']
        chk.loc[chk.index.isin(pred.index),['Type']] = "Test"
        chk['Pred'].fillna(chk['QtySold'], inplace=True)
        #print(label_test)
        mae = mean_absolute_error(label_test, pred)
        rmse = np.sqrt(mean_squared_error(label_test, pred['Predicted']))
        WAPE = np.sum(abs(label_test-pred['Predicted'])) / np.sum(label_test)
        WAPE = WAPE * 100
        np.mean(np.abs((label_test - pred['Predicted']) / label_test)) * 100
        MAPE = mape(label_test,pred['Predicted'])
        chk = post_prep_data(chk)
        f_pred = mdl.predict(chk_forecast)
        f_pred = pd.DataFrame(np.round(f_pred)).rename(columns={0:'Predicted'})
        f_pred.index = chk_forecast.index
        chk.loc[chk.index.isin(f_pred.index),['Pred']] = f_pred['Predicted']
        chk.loc[chk.index.isin(f_pred.index),['Type']] = "Forecasted"
        chk[['RMSE','WAPE','MAPE']] = rmse,WAPE,MAPE
        return chk
    else:
        return None
    
def model_fit_store(chk,end_date,forecast_date):
    #columns_to_fill = ['VendorID', 'VendorName', 'RetailerID', 'RetailerName']
    #print(np.unique(chk['str_sku_id']))
    #chk['Sales_7_Days_Lag'] = chk['proportion_sale'].shift(7)
    chk['Sales_7_Days_Lag'] = chk['QtySold'].shift(14)
    cols_selected = ['QtySold', 'Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday','Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday','Day_of_week_Wednesday',  'Holiday_0', 'Holiday_1','Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter','Sales_7_Days_Lag']#,'Inv_Avail',,'Inv_Avail_7_Days_Lag','Previousday_EOD_Inv','Inv_morn','tavg', 'wspd',
    #,'Inv_EOD_7_Days_Lag','Morn_Inv_7_Days_Lag'
    cols_to_drop = list(chk.columns[chk.isna().all()]) # finding columns that have all Nan values
    cols_selected = [col for col in cols_selected if col not in cols_to_drop] # removing columns that have all Nan values
    Q1 = chk['QtySold'].quantile(0.25)
    Q3 = chk['QtySold'].quantile(0.75)
    IQR = Q3-Q1
    avg_val = np.mean(chk['QtySold'][(chk['QtySold'] >= Q1-1.5*IQR) & (chk['QtySold'] <= Q3+1.5*IQR)])
    chk['QS_modified'] = np.where(((chk['QtySold'] <= Q1-1.5*IQR) | (chk['QtySold'] >= Q3+1.5*IQR)),avg_val,chk['QtySold'])
    cols_selected.append("QS_modified")
    chk['Type'] = "Train"
    #avg_sale = chk['QtySold'].mean()
    chk_train = chk[cols_selected][chk['ActualSaleDate']<=end_date]
    #chk_train['QtySold'] = np.where(chk_train['QtySold'] ==0,avg_sale,chk_train['QtySold'])
    chk_test = chk[cols_selected][(chk['ActualSaleDate']>end_date) & (chk['ActualSaleDate']<forecast_date)]
    #chk_test['QtySold'] = np.where(chk_test['QtySold'] ==0,avg_sale,chk_test['QtySold'])
    chk_forecast = chk[cols_selected][chk['ActualSaleDate']>=forecast_date]

    #Removing outliers and nans from train data
    chk_train.dropna(inplace=True)

    #chk_train = chk_train[(chk_train['QtySold'] >= Q1-1.5*IQR) & (chk_train['QtySold'] <= Q3+1.5*IQR)]
    label_train = chk_train.pop("QS_modified")
    label_test = chk_test.pop("QS_modified")
    label_forecast = chk_forecast.pop("QS_modified")
    chk_train.pop("QtySold"),chk_test.pop("QtySold"),chk_forecast.pop("QtySold")
    
    if(chk_train.shape[0]>chk_test.shape[0]):
        #Model fitting
        np.random.seed(475)
        rf = RandomForestRegressor()
        mdl = rf.fit(chk_train,label_train)

        #Prediction on test set
        pred = mdl.predict(chk_test)
        pred = pd.DataFrame(np.round(pred)).rename(columns={0:'Predicted'})

        #Formatting the output and calculating performance metrics
        pred.index = chk_test.index
        chk.loc[chk.index.isin(pred.index),['Pred']] = pred['Predicted']
        chk.loc[chk.index.isin(pred.index),['Type']] = "Test"
        chk['Pred'].fillna(chk['QtySold'], inplace=True)
        #mae = mean_absolute_error(label_test, pred)
        rmse = np.sqrt(mean_squared_error(label_test, pred['Predicted']))
        WAPE = np.sum(abs(label_test-pred['Predicted'])) / np.sum(label_test)
        WAPE = WAPE * 100
        MAPE = mape(label_test,pred['Predicted'])
        chk = post_prep_data(chk)
        f_pred = mdl.predict(chk_forecast)
        f_pred = pd.DataFrame(np.round(f_pred)).rename(columns={0:'Predicted'})
        f_pred.index = chk_forecast.index
        chk.loc[chk.index.isin(f_pred.index),['Pred']] = f_pred['Predicted']
        chk.loc[chk.index.isin(f_pred.index),['Type']] = "Forecasted"
        chk[['RMSE','WAPE','MAPE']] = rmse,WAPE,MAPE
        return chk
    else:
        return None
    
def test_data_sku(df):
    product_list = list(df['str_sku_id'].unique())
    #st.write(product_list)
    grouped_data = df.groupby('str_sku_id')
    
    # Create a list of DataFrames using dictionary comprehension
    #data_groups = [group_df.copy() for _, group_df in grouped_data]
    with st.spinner("Running model..."):
        # Display progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        product_output = []
        for i,product in enumerate(product_list):
            product_data = df[df['str_sku_id'] == product]
            #st.write(product_data['ProductID'].unique())
            end_date = np.max(product_data['ActualSaleDate'])-relativedelta(days=29)
            forecast_date = product_data['ActualSaleDate'].max() - relativedelta(days=6)

            #try:
            model_data_product = model_fit_SKU(product_data,end_date,forecast_date)
            product_output.append(model_data_product)

            #except Exception as e:
            #    st.error(f"Error processing product {product}: {str(e)}")
            #    continue

            # Update progress bar
            progress_bar.progress((i + 1) / len(product_list))
            progress_text.text(f"Progress: {i + 1} / {len(product_list)} products processed")

        product_output = pd.concat(product_output, ignore_index=True)
        #product_output = pd.DataFrame(product_output[0])
        st.session_state.Model_output_sku = product_output
        return product_output
    
def test_data_store(df):
    store_list = list(df['StoreID'].unique())
    #st.write(store_list)
    grouped_data = df.groupby('StoreID')
    with st.spinner("Running model..."):
        # Display progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        Store_output = []
        for i,store in enumerate(store_list):
            store_data = df[df['StoreID'] == store]
            end_date = np.max(store_data['ActualSaleDate'])-relativedelta(days=29)
            forecast_date = store_data['ActualSaleDate'].max() - relativedelta(days=6)

            #try:
            model_data_store = model_fit_store(store_data,end_date,forecast_date)
            Store_output.append(model_data_store)

            #except Exception as e:
            #    st.error(f"Error processing product {product}: {str(e)}")
            #    continue

            # Update progress bar
            progress_bar.progress((i + 1) / len(store_list))
            progress_text.text(f"Progress: {i + 1} / {len(store_list)} stores processed")

        Store_output = pd.concat(Store_output, ignore_index=True)
        #Store_output = pd.DataFrame(Store_output[0])
        st.session_state.Model_output_store = Store_output
        return Store_output

def draw_linechart(df):
    linechart = pd.DataFrame(df.groupby(df["ActualSaleDate"])[["QtySold","Pred"]].sum()).reset_index()
    mean_qty_sold = linechart["QtySold"].mean()
    fig2 = px.line(linechart, x = "ActualSaleDate", y=["QtySold","Pred"], labels = {"value": "Qty","SKU":"ProductID"},height=500, width = 1000,color_discrete_map={"QtySold": "green", "Pred": "blue"})#,template="gridon",hover_data=[linechart["ProductID"],linechart[ "StoreID"]]

    # Adding mean lines to the chart
    fig2.add_hline(y=mean_qty_sold, line_dash="dot", line_color="#FFBF00", annotation_text=f'Mean QtySold: {mean_qty_sold:.2f}', annotation_position="bottom right")
    #fig2 = fig2.update_traces(hovertemplate=df["ProductID"])
    #st.plotly_chart(fig2,use_container_width=True)
    return fig2  

def visualize_sku_level():
    #st.session_state.visualize = True
    # Filter the data based on selected Store and SKU
    f_cast_sku = st.session_state.Model_output_sku
    #st.write(f_cast_sku['ProductID'].unique())
    st.header("Choose your filters for store-sku level forecast: ")
    # Create filters for Store and SKU
    #fl1, fl2 = st.columns((2))
    #with fl1:
    Store_filter = st.multiselect("Pick your Store", f_cast_sku["StoreID"].unique())
    #with fl2:
    if not Store_filter:
        p_df1_sku = f_cast_sku[f_cast_sku['Type']=="Test"].copy()
        #chart = draw_linechart(p_df1_sku)
        #st.plotly_chart(chart,use_container_width=True)
    else:
        p_df1_sku =f_cast_sku[(f_cast_sku["StoreID"].isin(Store_filter)) & (f_cast_sku['Type']=="Test")]
        #chart = draw_linechart(p_df1_sku)
        #st.plotly_chart(chart,use_container_width=True)
    SKU_filter = st.multiselect("Pick your SKU", p_df1_sku["ProductID"].unique())
    
    if not SKU_filter:
        p_df2 = p_df1_sku.copy()
        #accuracy = ((np.round(p_df2['WAPE'].unique()[0],2)))
        error = ((np.round(np.median(p_df2['WAPE']),2)))
        chart = draw_linechart(p_df2)
        st.plotly_chart(chart,use_container_width=True)
        st.write(f"Error rate (variation of predicted to actual sales): {error}%")
    else:
        p_df2 =p_df1_sku[p_df1_sku["ProductID"].isin(SKU_filter)] #& (p_df1['Type']=='Test')
        #accuracy = ((np.round(p_df2['WAPE'].unique()[0],2)))
        error = ((np.round(np.median(p_df2['WAPE']),2)))
        #st.write(p_df2)
        chart = draw_linechart(p_df2)
        st.plotly_chart(chart,use_container_width=True)
        st.write(f"Error rate (variation of predicted to actual sales): {error}%")

def visualize_store_level():
    #st.session_state.visualize = True
    # Filter the data based on selected Store and SKU
    f_cast_store = st.session_state.Model_output_store
    st.header("Choose your filters for store level forecast: ")
    # Create filters for Store and SKU
    #fl1, fl2 = st.columns((2))
    #with fl1:
    Store_filter1 = st.multiselect("Pick your Store", f_cast_store["StoreID"].unique(),key="store_level_filter")
    #with fl2:
    if not Store_filter1:
        p_df1_store = f_cast_store[f_cast_store['Type']=="Test"].copy()
        chart = draw_linechart(p_df1_store)
        st.plotly_chart(chart,use_container_width=True)
        error = ((np.round(np.median(p_df1_store['WAPE']),2)))
        st.write(f"Error rate (variation of predicted to actual sales): {error}%")
    else:
        p_df1_store =f_cast_store[(f_cast_store["StoreID"].isin(Store_filter1)) & (f_cast_store['Type']=="Test")]
        chart = draw_linechart(p_df1_store)
        st.plotly_chart(chart,use_container_width=True)
        error = ((np.round(np.median(p_df1_store['WAPE']),2)))
        st.write(f"Error rate (variation of predicted to actual sales): {error}%")

#def visualize_store_level():
def app(df_sku,df_store,Test_state):#,skudata_button_state,storedata_button_state):
    #st.write(Test_state)
    #st.write(st.session_state.Model_output_sku)
    #st.write(st.session_state.Model_output_store)
    #st.session_state.skudata_button_state=skudata_button_state
    #st.session_state.storedata_button_state=storedata_button_state
    #if st.button("Test") or Test_state:
    if st.session_state.Model_output_sku is None and st.session_state.Model_output_store is None:
        st.session_state.Test_button_state = True
        t_cast_sku = test_data_sku(df_sku)
        st.session_state.test_completed_sku = "True"
        t_cast_store = test_data_store(df_store)
        st.session_state.test_completed_store = "True"

        f_cast_sku = t_cast_sku[(t_cast_sku['Type']=="Train") | (t_cast_sku['Type']=="Test")]
        f_cast_store = t_cast_store[(t_cast_store['Type']=="Train") | (t_cast_store['Type']=="Test")]
        #if st.session_state.skudata_button or st.session_state.storedata_button:
        col1, col2 = st.columns((2))
        with col1:
            skudata_button = st.button("Store-SKU Level Predictions", key="skudata_button")
            #st.session_state.skudata_button = True
        with col2:
            storedata_button = st.button("Store Level Predictions", key="storedata_button")
            #st.session_state.storedata_button = True
        if skudata_button or st.session_state.skudata_button_state:
            m_numRows = f_cast_sku.shape[0]
            st.header("Store-SKU Level Predictions")
            st.write("Prediction at SKU level is done by training the model on all except last 1 month data. Last 1 month of data is used to test the model")
            st.dataframe(f_cast_sku)#,height =(m_numRows + 1) * 35 + 3,hide_index=True            
            st.session_state.Model_output_sku = t_cast_sku
            st.session_state.skudata_button_state = True
            st.session_state.storedata_button_state = False
            visualize_sku_level()

        if storedata_button or st.session_state.storedata_button_state:
            st.session_state.skudata_button_state = False
            st.session_state.button_click_count += 1
            st.header("Store Level Predictions")
            st.write("Prediction at Store level is done by training the model on all except last 1 month data. Last 1 month of data is used to test the model")
            st.dataframe(f_cast_store)#,height =(m_numRows + 1) * 35 + 3,hide_index=True
            st.session_state.Model_output_store = t_cast_store
            st.session_state.storedata_button_state = True
            visualize_store_level()

    else:
        #if st.session_state.skudata_button or st.session_state.storedata_button:
        col1, col2 = st.columns((2))
        with col1:
            skudata_button = st.button("Store-SKU Level Predictions", key="skudata_button")
        with col2:
            storedata_button = st.button("Store Level Predictions", key="storedata_button")
        if skudata_button or st.session_state.skudata_button_state:
            st.header("Store-SKU Level Predictions")
            st.write("Prediction at SKU level is done by training the model on all except last 1 month data. Last 1 month of data is used to test the model")
            t_cast_sku = st.session_state.Model_output_sku
            #st.write(t_cast_sku['ProductID'].unique())
            f_cast_sku = t_cast_sku[(t_cast_sku['Type']=="Train") | (t_cast_sku['Type']=="Test")]
            st.dataframe(f_cast_sku)#,hide_index=True
            st.session_state.test_completed_sku = True
            st.session_state.skudata_button_state = True
            st.session_state.storedata_button_state = False
            visualize_sku_level()

        if storedata_button or st.session_state.storedata_button_state:
            st.session_state.skudata_button_state = False
            storedata_button
            st.header("Store Level Predictions")
            st.write("Prediction at Store level is done by training the model on all except last 1 month data. Last 1 month of data is used to test the model")
            t_cast_store = st.session_state.Model_output_store
            f_cast_store = t_cast_store[(t_cast_store['Type']=="Train") | (t_cast_store['Type']=="Test")]
            st.dataframe(f_cast_store)#,hide_index=True
            st.session_state.test_completed_store = True
            st.session_state.storedata_button_state = True
            visualize_store_level()
