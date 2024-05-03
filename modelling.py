import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from dateutil.relativedelta import relativedelta
#from statsmodels.tools.eval_measures import rmse

#end_date = pd.to_datetime('2022-06-30')


def mape(actual, pred): 
    '''
    Mean Absolute Percentage Error (MAPE) Function
    
    input: list/series for actual values and predicted values
    output: mape value 
    '''
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

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
    df["ActualSaleDate"] = pd.to_datetime(df["ActualSaleDate"],format='%d-%m-%Y')# '%Y-%m-%d'
    f_df = prepare_future_data(df,7)
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

def prepare_future_data(p_df,num_of_days):
    unique_dates = p_df['ActualSaleDate'].unique()
    sorted_dates = sorted(unique_dates, reverse=True)
    last_x_dates = sorted_dates[:num_of_days]
    f_df = p_df[['StoreID','ProductID','Inv_Avail','storeCity','storeState','StoreZip5']][p_df['ActualSaleDate'].isin(last_x_dates)]
    f_start_date = p_df['ActualSaleDate'].max() + relativedelta(days=1)
    f_end_date = p_df['ActualSaleDate'].max() + relativedelta(days=num_of_days)
    f_df['ActualSaleDate'] = pd.date_range(f_start_date,f_end_date)
    return f_df

def post_prep_data(chk):
    one_hot_columns = [col for col in chk.columns if col.startswith(('Day_of_week_', 'Holiday','Season_'))]
    chk['DAYOFWEEK_NM'] = (chk[[col for col in chk if col.startswith(('Day_of_week_'))]]==1).idxmax(1)
    chk['Holiday'] = (chk[[col for col in chk if col.startswith(('Holiday_'))]]==1).idxmax(1)
    chk['Season'] = (chk[[col for col in chk if col.startswith(('Season_'))]]==1).idxmax(1)
    chk['DAYOFWEEK_NM'] = chk['DAYOFWEEK_NM'].str.replace('Day_of_week_','')
    chk['Holiday'] = chk['Holiday'].str.replace('Holiday_','')
    chk['Season'] = chk['Season'].str.replace('Season_','')
    lag_columns = [col for col in chk.columns if 'Lag' in col]
    chk = chk.drop(columns=one_hot_columns+lag_columns)
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


def model_fit(chk,end_date,forecast_date):
    #columns_to_fill = ['VendorID', 'VendorName', 'RetailerID', 'RetailerName']
    #print(np.unique(chk['str_sku_id']))
    #chk['Sales_7_Days_Lag'] = chk['proportion_sale'].shift(7)
    chk['Sales_7_Days_Lag'] = chk['QtySold'].shift(7)
    chk['Inv_Avail_7_Days_Lag'] = chk['Inv_Avail'].shift(7)
    #chk['Previousday_Morn_Inv'] = chk['Inv_morn'].shift(1)
    #chk['Morn_Inv_7_Days_Lag'] = chk['Inv_morn'].shift(7)
    #chk['Previousday_EOD_Inv'] = chk['Inv_eod'].shift(1)
    cols_selected = ['QtySold', 'Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday','Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday','Day_of_week_Wednesday',  'Holiday_0', 'Holiday_1','Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter','Sales_7_Days_Lag','Inv_Avail_7_Days_Lag']#,'Inv_Avail','Previousday_EOD_Inv','Inv_morn','tavg', 'wspd',
    #,'Inv_EOD_7_Days_Lag','Morn_Inv_7_Days_Lag'
    cols_to_drop = list(chk.columns[chk.isna().all()]) # finding columns that have all Nan values
    cols_selected = [col for col in cols_selected if col not in cols_to_drop] # removing columns that have all Nan values
    chk['Type'] = "Train"
    chk_train = chk[cols_selected][chk['ActualSaleDate']<=end_date]
    chk_test = chk[cols_selected][(chk['ActualSaleDate']>end_date) & (chk['ActualSaleDate']<forecast_date)]
    chk_forecast = chk[cols_selected][chk['ActualSaleDate']>=forecast_date]

    #Removing outliers and nans from train data
    chk_train.dropna(inplace=True)
    Q1 = chk_train['QtySold'].quantile(0.25)
    Q3 = chk_train['QtySold'].quantile(0.75)
    IQR = Q3-Q1
    chk_train = chk_train[(chk_train['QtySold'] >= Q1-1.5*IQR) & (chk_train['QtySold'] <= Q3+1.5*IQR)]
    label_train = chk_train.pop("QtySold")
    label_test = chk_test.pop("QtySold")
    label_forecast = chk_forecast.pop("QtySold")
    
    if(chk_train.shape[0]>chk_test.shape[0]) :
       #and (len(label_train>0)>3) and (len(label_test>0)>3))
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
        print(label_test)
        mae = mean_absolute_error(label_test, pred)
        rmse = np.sqrt(mean_squared_error(label_test, pred['Predicted']))
        WAPE = np.sum(abs(label_test-pred['Predicted'])) / np.sum(label_test)
        np.mean(np.abs((label_test - pred['Predicted']) / label_test)) * 100
        MAPE = mape(label_test,pred['Predicted'])
        chk = post_prep_data(chk)
        f_pred = mdl.predict(chk_forecast)
        f_pred = pd.DataFrame(np.round(f_pred)).rename(columns={0:'Predicted'})
        f_pred.index = chk_forecast.index
        chk.loc[chk.index.isin(f_pred.index),['Pred']] = f_pred['Predicted']
        chk.loc[chk.index.isin(f_pred.index),['Type']] = "Forecasted"
        chk[['RMSE','MAE','WAPE','MAPE']] = rmse,mae,WAPE,MAPE
        return chk
    else:
        return None