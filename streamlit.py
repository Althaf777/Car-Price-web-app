import streamlit as st
import pickle
import pandas as pd
import numpy as np

reg_model=pickle.load(open('rid.pkl','rb'))
scale_model=pickle.load(open('scaler.pkl','rb'))

st.title("Car Price Prediction")

df=pd.read_csv(r"C:\Users\AL\Downloads\udemy ML\Car projects\datasets\car data.csv")
df1=df.drop('Car_Name',axis=1)
df1['Total_years']=2024-df1['Year']
df2=df1.drop('Year',axis=1)


st.subheader("Input Features")

Present_Price=st.number_input("Present Price(in Lakhs)", step=0.01)
Total_years=st.slider("How many years old",0,40)
Fuel_Type=st.selectbox('Fuel Type',df['Fuel_Type'].unique())
Kms_Driven=st.number_input("Kms driven",11,200000)
Seller_Type=st.selectbox('Seller Type',df['Seller_Type'].unique())
Transmission=st.selectbox('Transmission',df['Transmission'].unique())
Owner=st.selectbox('Owner',df['Owner'].unique())

if st.button("Predict"):
    input=pd.DataFrame([[Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner,Total_years]],columns=['Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner','Total_years'])
    
    input.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

    #encoding 'Seller_type column'
    input.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

    #encoding 'Transmission'
    input.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

    input2=scale_model.transform(input)
    car_price=reg_model.predict(input2)

    st.markdown('The price of the car will be  '  + str(car_price[0]))



