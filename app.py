import pandas as pd
import numpy as np
import streamlit as st 
import joblib
import pickle

model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

st.title("Water Pollutants predictor")
st.write("Predict the water pollutants based on yeat and Station Id.")

year_ip = st.number_input("Enter year",min_value=2000,max_value=2100,value = 2022)
station_id = st.text_input("Enter Station Id",value='1')

if st.button('predict'):
    if not station_id:
        st.write("Enter the station id")
    else:
        input_df=pd.DataFrame({'year':[year_ip], 'id':[station_id]})
        input_encoded=pd.get_dummies(input_df,columns=['id'])
        
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded=input_encoded[model_cols]
        
        predicted_poll = model.predict(input_encoded)[0]
        pollutants = ['O2','NO3','NO2','SO4','PO4','CL']
        
        st.subheader(f"predict pollutants level for the station'{station_id}'in {year_ip}:")
        predicted_values={}
        for p,val in zip(pollutants,predicted_poll):
            st.write(f'{p}:{val:.2f}')
                
        
    
