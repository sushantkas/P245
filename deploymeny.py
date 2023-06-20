import pandas as pd
import numpy as np
import streamlit as st
import pickle


st.title("Welcome to check Customer segmentation")

image="https://media.istockphoto.com/id/1336161711/vector/hand-holding-magifier-to-find-target-audience-among-other-people-on-grey-background.jpg?s=1024x1024&w=is&k=20&c=414XvgXgE5OGvMukfZGKCq8WfotroRe-ptEjGKuI5y0="

def add_bg_from_url(link):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({link});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url(image)


# Loading scaler and Encoder files

with open("Stadnard_scaler.pkl", "rb") as file1:
    scaler=pickle.load(file1)

# Label encdoer 
with open("edu_encoder.pkl","rb") as file2:
    edu_encoder=pickle.load(file2)

with open("marit_encoder.pkl", "rb") as file3:
    label_encoder=pickle.load(file3)


with open("svm_classifier.pkl", "rb") as file:
    svm=pickle.load(file)




# Education (int32)
education = st.selectbox("Education", ("Graduation","PhD","Master","2n Cycle","Basic"))

# Marital_Status (int32)
marital_status = st.selectbox("Marital Status", ("Married","Together","Single","Divorced","Widow","Alone","Absurd","YOLO"))

# Income (float64)
income = st.number_input("Income", min_value=2000.0, step=1000.0)

# Kidhome (int64)
kidhome = st.number_input("Kidhome", value=0, step=1, max_value=3, min_value=0)

# Teenhome (int64)
teenhome = st.number_input("Teenhome", value=0, step=1, max_value=2, min_value=0)

# Recency (int64)
recency = st.number_input("Recency", value=0, step=1,max_value=56, min_value=0)

# MntWines (float64)
mnt_wines = st.number_input("MntWines", value=0.0, step=1.0, max_value=1500.0, min_value=0.0)

# MntFruits (float64)
mnt_fruits = st.number_input("MntFruits", value=33.0, step=1.0, max_value=100.0, min_value=0.0)

# MntMeatProducts (float64)
mnt_meat_products = st.number_input("MntMeatProducts", value=232.0, step=1.0, max_value=2000.0, min_value=0.0)

# MntFishProducts (float64)
mnt_fish_products = st.number_input("MntFishProducts", value=50.0, step=1.0, max_value=259.0, min_value=0.0)

# MntSweetProducts (float64)
mnt_sweet_products = st.number_input("MntSweetProducts", value=33.0, step=1.0, max_value=270.0, min_value=0.0)

# MntGoldProds (float64)
mnt_gold_prods = st.number_input("MntGoldProds", value=56.0, step=1.0, max_value=400.0, min_value=0.0)

# NumDealsPurchases (int64)
num_deals_purchases = st.number_input("NumDealsPurchases", value=7, step=1, max_value=15, min_value=0)

# NumWebPurchases (int64)
num_web_purchases = st.number_input("NumWebPurchases", value=6, step=1, max_value=30, min_value=0)

# NumCatalogPurchases (int64)
num_catalog_purchases = st.number_input("NumCatalogPurchases", value=4, step=1, max_value=30, min_value=0)

# NumStorePurchases (int64)
num_store_purchases = st.number_input("NumStorePurchases", value=5, step=1, max_value=15, min_value=0)

# NumWebVisitsMonth (int64)
num_web_visits_month = st.number_input("NumWebVisitsMonth", value=7, step=1, max_value=30, min_value=0)

# AcceptedCmp1 (int64)
accepted_cmp1 = st.number_input("AcceptedCmp1", value=0, step=1,max_value=1)

# AcceptedCmp2 (int64)
accepted_cmp2 = st.number_input("AcceptedCmp2", value=0, step=1,max_value=1)

# AcceptedCmp3 (int64)
accepted_cmp3 = st.number_input("AcceptedCmp3", value=0, step=1,max_value=1)

# AcceptedCmp4 (int64)
accepted_cmp4 = st.number_input("AcceptedCmp4", value=0, step=1,max_value=1)

# AcceptedCmp5 (int64)
accepted_cmp5 = st.number_input("AcceptedCmp5", value=0, step=1,max_value=1)

# Complain (int64)
complain = st.number_input("Complain", value=0, step=1,max_value=1)

# Response (int64)
response = st.number_input("Response", value=0, step=1,max_value=1)


data=pd.DataFrame({'Education':education, 'Marital_Status':marital_status, 'Income':income, 'Kidhome':kidhome, 'Teenhome':teenhome,
       'Recency':recency, 'MntWines':mnt_wines, 'MntFruits':mnt_fruits, 'MntMeatProducts':mnt_meat_products,
       'MntFishProducts':mnt_fish_products, 'MntSweetProducts':mnt_sweet_products, 'MntGoldProds':mnt_gold_prods,
       'NumDealsPurchases':num_deals_purchases, 'NumWebPurchases':num_web_purchases, 'NumCatalogPurchases':num_catalog_purchases,
       'NumStorePurchases':num_store_purchases, 'NumWebVisitsMonth':num_web_visits_month, 'AcceptedCmp3':accepted_cmp3,
       'AcceptedCmp4':accepted_cmp4, 'AcceptedCmp5':accepted_cmp5, 'AcceptedCmp1':accepted_cmp1, 'AcceptedCmp2':accepted_cmp2,
       'Complain':complain, 'Response':response}, index=["0"])

st.dataframe(data)

continuous_column=["Income","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]


if st.button("Check Customer Segment"):
    st.write("Encoding data for Algorithm")
    data[continuous_column]=scaler.transform(data[continuous_column])
    data["Education"]=edu_encoder.transform(data["Education"])
    data["Marital_Status"]=label_encoder.transform(data["Marital_Status"])
    st.write("Encoded Successfully")
    cust_class=svm.predict(data)[0]
    st.success(f"Customer belongs to Class {cust_class+1}")
    st.snow()
else:
    st.stop()