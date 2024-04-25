import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

end_date = pd.to_datetime('2022-06-30')
def model_fit(chk):
    #columns_to_fill = ['VendorID', 'VendorName', 'RetailerID', 'RetailerName']
    #print(np.unique(chk['str_sku_id']))
    #chk['Sales_7_Days_Lag'] = chk['proportion_sale'].shift(7)
    chk['Sales_7_Days_Lag'] = chk['QtySold'].shift(7)
    #chk['Inv_EOD_7_Days_Lag'] = chk['Inv_eod'].shift(7)
    #chk['Previousday_Morn_Inv'] = chk['Inv_morn'].shift(1)
    #chk['Morn_Inv_7_Days_Lag'] = chk['Inv_morn'].shift(7)
    chk['Previousday_EOD_Inv'] = chk['Inv_eod'].shift(1)
    cols_selected = ['QtySold','tavg', 'wspd', 'Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday','Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday','Day_of_week_Wednesday',  'Holiday_0', 'Holiday_1','Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter','Sales_7_Days_Lag']#,'Previousday_EOD_Inv','Inv_morn'
    #,'Inv_EOD_7_Days_Lag','Morn_Inv_7_Days_Lag'
    cols_to_drop = list(chk.columns[chk.isna().all()]) # finding columns that have all Nan values
    cols_selected = [col for col in cols_selected if col not in cols_to_drop] # removing columns that have all Nan values
    chk_train = chk[cols_selected][chk['ActualSaleDate']<=end_date]
    chk_test = chk[cols_selected][chk['ActualSaleDate']>end_date]

    #Removing outliers and nans from train data
    chk_train.dropna(inplace=True)
    Q1 = chk_train['QtySold'].quantile(0.25)
    Q3 = chk_train['QtySold'].quantile(0.75)
    IQR = Q3-Q1
    chk_train = chk_train[(chk_train['QtySold'] >= Q1-1.5*IQR) & (chk_train['QtySold'] <= Q3+1.5*IQR)]
    label_train = chk_train.pop("QtySold")
    label_test = chk_test.pop("QtySold")
    
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
        chk['Pred'].fillna(chk['QtySold'], inplace=True)
        mae = mean_absolute_error(label_test, pred)
        rmse = np.sqrt(mean_squared_error(label_test, pred['Predicted']))
        WMAPE = np.sum(abs(label_test-pred['Predicted'])) / np.sum(label_test)
        chk[['RMSE','MAE','WMAPE']] = rmse,mae,WMAPE
        one_hot_columns = [col for col in chk.columns if col.startswith(('Day_of_week_', 'Holiday','Season_'))]
        chk['DAYOFWEEK_NM'] = (chk[[col for col in chk if col.startswith(('Day_of_week_'))]]==1).idxmax(1)
        chk['Holiday'] = (chk[[col for col in chk if col.startswith(('Holiday_'))]]==1).idxmax(1)
        chk['Season'] = (chk[[col for col in chk if col.startswith(('Season_'))]]==1).idxmax(1)
        chk['DAYOFWEEK_NM'] = chk['DAYOFWEEK_NM'].str.replace('Day_of_week_','')
        chk['Holiday'] = chk['Holiday'].str.replace('Holiday_','')
        chk['Season'] = chk['Season'].str.replace('Season_','')
        lag_columns = [col for col in chk.columns if 'Lag' in col]
        chk = chk.drop(columns=one_hot_columns+lag_columns)
        chk['zero_sales_streak'] = (chk['QtySold'] == 0).astype(int)

        # Use the cumsum() function to assign a unique group number to consecutive zero sales streaks
        chk['zero_sales_group'] = chk['zero_sales_streak'].cumsum()
        # Use the cumsum() function to assign a unique group number to consecutive zero sales streaks
        chk['zero_sales_group'] = (chk['QtySold'] != 0).astype(int).cumsum()
        # Calculate the count of consecutive zero sales days for each group
        chk['consecutive_zero_sales'] = chk.groupby('zero_sales_group')['zero_sales_streak'].cumsum()
        # Filter out the rows where sales are not zero
        chk['consecutive_zero_sales'] = chk['consecutive_zero_sales'] * (chk['QtySold'] == 0)
        # Drop the intermediate columns if needed
        chk = chk.drop(['zero_sales_streak', 'zero_sales_group'], axis=1)
        return chk
    else:
        return None