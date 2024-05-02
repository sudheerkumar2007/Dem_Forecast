import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

def draw_linechart(df):
    linechart = pd.DataFrame(df.groupby(df["ActualSaleDate"])[["QtySold","Pred"]].sum()).reset_index()
    fig2 = px.line(linechart, x = "ActualSaleDate", y=["QtySold","Pred"], labels = {"value": "Qty","SKU":"ProductID"},height=500, width = 1000,template="gridon")#hover_data=[linechart["ProductID"],linechart[ "StoreID"]]
    #fig2 = fig2.update_traces(hovertemplate=df["ProductID"])
    #st.plotly_chart(fig2,use_container_width=True)
    return fig2

def app():
    visualize_op_button = st.button("Visualize", key="visualize_op_button")
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
                p_df2 = p_df1[p_df1['Type']=='Test']
                accuracy = 100-(np.unique(p_df2['WAPE'])*100)
                st.write(f"Results displayed are {accuracy}% accurate")
                chart = draw_linechart(p_df2)
                st.plotly_chart(chart,use_container_width=True)
            else:
                p_df2 =p_df1[p_df1["ProductID"].isin(SKU_filter)]
                accuracy = 100-(np.unique(p_df2['WAPE'])*100)
                st.write(f"Results displayed are {accuracy}% accurate")
                chart = draw_linechart(p_df2)
                st.plotly_chart(chart,use_container_width=True)
