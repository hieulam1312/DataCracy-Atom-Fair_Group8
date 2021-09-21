from numpy import histogram
from numpy.lib.function_base import append
import streamlit as st
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials #-> Để nhập Google Spreadsheet Credentials
import gspread
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive'],
)
gc1 = gspread.authorize(credentials)
spreadsheet_key = '1lAlQLlUBaqCL1Onh0JfK9pZGEbzqSCupCPo1r3UGj6c'



sh1=gc1.open("Kho mẫu - Test").worksheet('Sheet1')
sample_name=sh1.get_all_records()
sample_name_pd=pd.DataFrame(sample_name)
step=st.selectbox('Chọn thao tác',['Trả mẫu','Mượn mẫu'])

filter=st.multiselect('Chọn Khách hàng',sample_name_pd['TÊN KHÁCH HÀNG'].unique().tolist())
sample=sample_name_pd[sample_name_pd['TÊN KHÁCH HÀNG'].isin(filter)]

sp=st.multiselect('Chọn sản phẩm',sample['TÊN SẢN PHẨM'].unique())
table=sample[sample['TÊN SẢN PHẨM'].isin(sp)]
table
# sub_folders=order_df[['TÊN KHÁCH HÀNG','TÊN SẢN PHẨM']]
# sub_folders.set_index('TÊN KHÁCH HÀNG').T.to_dict('list')
