import pandas as pd
import streamlit as st
import plotly.express as px
from tqdm import tqdm
from modelling import model_fit,get_processed_df
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Demand Forecasting app",page_icon=":chart_with_upwards_trend:")
#st.beta_set_page_config( layout='wide',page_title="Demand Forecasting app",page_icon=":chart_with_upwards_trend:")
st.title(" :chart_with_upwards_trend: Business Demand Forecast")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)


def display_data(p_df, date1, date2):
        df1 = p_df[(p_df["ActualSaleDate"] >= date1) & (p_df["ActualSaleDate"] <= date2)].copy().sort_values(by="ActualSaleDate")
        numRows = df1.shape[0]
        numcols = df1.shape[1]
        df_height = (numRows + 1) #* 35 + 3
        df_width = numcols +1
        st.write("Your Data looks like this:")
        #st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        return st.dataframe(df1,use_container_width = True,height = df_height, width = df_width)#hide_index=True,

def test_data(df):
    product_list = list(df['str_sku_id'].unique())
    grouped_data = df.groupby('str_sku_id')
    end_date = np.max(df['ActualSaleDate'])-relativedelta(days=29)
    forecast_date = df['ActualSaleDate'].max() - relativedelta(days=6)
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
            product_data = model_fit(product_data,end_date,forecast_date)
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
        #print(model_output.columns)
        #model_output['MAPE'] = model_output['MAPE'].replace([np.inf, -np.inf], 100)
        #model_output['Model'] = 'RF_with_Season_No_invmorn_previnv_outliers_removed'
        return model_output

def draw_linechart(df):
    linechart = pd.DataFrame(df.groupby(df["ActualSaleDate"])[["QtySold","Pred"]].sum()).reset_index()
    mean_qty_sold = linechart["QtySold"].mean()
    fig2 = px.line(linechart, x = "ActualSaleDate", y=["QtySold","Pred"], labels = {"value": "Qty","SKU":"ProductID"},height=500, width = 1000)#,template="gridon",hover_data=[linechart["ProductID"],linechart[ "StoreID"]]

    # Adding mean lines to the chart
    fig2.add_hline(y=mean_qty_sold, line_dash="dot", line_color="red", annotation_text=f'Mean QtySold: {mean_qty_sold:.2f}', annotation_position="bottom right")
    #fig2 = fig2.update_traces(hovertemplate=df["ProductID"])
    #st.plotly_chart(fig2,use_container_width=True)
    return fig2  

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
    if "test_completed" not in st.session_state:
        st.session_state.test_completed = False
    if "Forecast_state" not in st.session_state:
        st.session_state.Forecast_state = False
    if "Test_button_state" not in st.session_state:
        st.session_state.Test_button_state = False
    if "Model_output" not in st.session_state:
        st.session_state.Model_output = None
    if "visualize" not in st.session_state:
        st.session_state.visualize = False
    if "predict" not in st.session_state:
        st.session_state.predict_state = False
    if "t_cast" not in st.session_state:
        st.session_state.tcast = None

    forecast_template = pd.read_csv("DemandForecast_template.csv")  
    forecast_template_str = forecast_template.to_csv(index=False)  
    st.sidebar.download_button(label="Download Sales data template",
                               data=forecast_template_str,
                               file_name='forecast_template.csv',
                               mime='text/csv')    

    #st.sidebar.write(button_html.format(label="Download Sales data template", data=forecast_template_str, file_name='forecast_template.csv'))
    st.sidebar.subheader("Your dataset")
    file = st.sidebar.file_uploader("upload your file here",type={"csv"})

    if st.sidebar.button("Preprocess") or st.session_state.Next_state:
        st.session_state.Next_state = True
        with st.spinner("processing"):
            #Read the data
            df = pd.read_csv(file)
            #Preprocess the dataframe
            p_df = get_processed_df(df)
            #st.write("Preprocessing complete, You can view/Forecast data now")

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
                    view_data_button = st.button("View Data", key="display_data_button")
                with cl2:
                    Dashboard_button = st.button("Sales Dashboard", key="Dashboard_button")
                with cl3:
                    Test_button = st.button("Test", key="Test_button")
                #with cl3:
                #    visualize_op_button = st.button("Visualize output",key = "visualize_op_button")
                #with cl4:
                #    predict_button = st.button("Predict",key = "predict_button")

                #Displaying data
                if view_data_button:
                    with st.spinner("Displaying"):
                        display_data(p_df,st.session_state.date1,st.session_state.date2)

                #Forecasting
                if Test_button or st.session_state.Test_button_state:
                    if st.session_state.Model_output is None:
                        st.session_state.Test_button_state = True
                        t_cast = test_data(p_df)
                        st.session_state.tcast = t_cast
                        f_cast = t_cast[(t_cast['Type']=="Train") | (t_cast['Type']=="Test")]
                        m_numRows = f_cast.shape[0]
                        st.write("Forecast is done by training the model on all except last 1 month data. Last 1 month of data is used to test the model. Here is the output")
                        st.dataframe(f_cast)#,height =(m_numRows + 1) * 35 + 3,hide_index=True
                        st.session_state.test_completed = "True"
                        st.session_state.Model_output = f_cast
                    else:
                        st.write("Forecast is already complete. It is done by training the model on all except last 1 month data. Last 1 month of data is used to test the model. Here is the output")
                        f_cast = st.session_state.Model_output
                        st.dataframe(f_cast)#,hide_index=True

                #Visualizing output
                #if visualize_op_button or st.session_state.visualize:
                    if not st.session_state.test_completed:
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
                            p_df2 = p_df1[p_df1['Type']=='Test']
                            accuracy = 100-((np.round(p_df2['WAPE'].unique()[0],2))*100)
                            #st.write(f"Results displayed are {accuracy}% accurate")
                            chart = draw_linechart(p_df2)
                            st.plotly_chart(chart,use_container_width=True)
                        else:
                            p_df2 =p_df1[(p_df1["ProductID"].isin(SKU_filter)) & (p_df1['Type']=='Test')]
                            accuracy = 100-((np.round(p_df2['WAPE'].unique()[0],2))*100)
                            #st.write(f"Results displayed are {accuracy}% accurate")
                            chart = draw_linechart(p_df2)
                            st.plotly_chart(chart,use_container_width=True)

            if st.sidebar.button("Forecast") or st.session_state.Forecast_state:
                st.session_state.Forecast_state = "True"
                st.session_state.Next_state = "False"
                #print(p_df.shape)
                t_cast = test_data(p_df)
                cols = ['ActualSaleDate','DAYOFWEEK_NM','Season','StoreID','ProductID','storeCity','storeState','StoreZip5','Pred']
                f_df = t_cast[cols][t_cast['Type']=="Forecasted"].rename(columns={"Pred":"Predicted_Sale_Qty"})
                st.write("Here is the forecasted sale for next 7 days")
                st.dataframe(f_df)#,height =(m_numRows + 1) * 35 + 3,hide_index=True
                st.session_state.f_op = f_cast
                accuracy = 100-((np.round(t_cast['WAPE'].unique()[0],2))*100)
                st.write(f"Forecast done is {accuracy}% accurate")

    
if __name__ == '__main__':
	main()