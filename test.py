import streamlit as st
import numpy as np
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq
from googleapiclient.discovery import build
from google.cloud import bigquery
import json
#import mysql.connector
import os
from PIL import Image
import streamlit.components.v1 as components
import streamlit.components.v1 as stc 
import codecs
import sys

# Create API client.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets['gcp_service_account']
# )
# client = bigquery.Client(credentials=credentials)

#Streamlit app
st.set_page_config(layout="wide")

st.title('HANDPICK CONCEPT COMPANY LIMITED')
st.subheader('Data Warehouse - Google BigQuery')
#image = Image.open('C:/Users/PC/Pictures/picture 3.png')
#st.image(image, caption= 'Design by HandPickConcept', use_column_width=True)

st.sidebar.header('HOME')
#image = Image.open('C:/Users/PC/Pictures/picture4.jpg')
#st.sidebar.image(image,width=300)
st.sidebar.subheader('About HANDPICK Concept')
st.sidebar.write('**HANDPICK CONCEPT CO.,LTD**')
st.sidebar.write('📌**Address**: S1 The Sun Avenue Tower, 28 Mai Chi Tho, An Phu Ward, Thu Duc, Ho Chi Minh City, Vietnam')
st.sidebar.write('☎️ **Contact**: xxxxxxxxxx')

st.subheader('HANDPICK DATA')

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def normalize_db_table_column(df):
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace(r'\n', '_')
    df.columns = df.columns.str.replace(r'[^\w\s]+', '_')
    df.columns = df.columns.str.lower()
    df.columns = df.columns.to_series().apply(remove_accents)


#Setup push BigQuery
def push_exit_table(df, db_table):
    normalize_db_table_column(df)
    print(df)
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials)

    project_id="hp-data-324704"

    pandas_gbq.to_gbq(df,f'DWhandpick.{db_table}', project_id=project_id, if_exists='append')

    st.write("Please upload an Excel or CSV file")


#checkbox

if st.checkbox("Import Files"):
    db_table = st.selectbox("📍 Database Table 📍", ["Customer", "DO", "GO", "Inventory", "Product", "Production", "Return_SO", "SO", "Sup_Product"])
    product_file = st.file_uploader(label="📤 Before selecting the file you want to import, please choosing 'Database Table' first 👆...", type = 'xlsx')
    if product_file:
        print('Process: ', product_file, db_table)
        df1= pd.read_excel(product_file, engine = 'openpyxl')
        st.dataframe(df1)
        push_exit_table(df1, db_table)
