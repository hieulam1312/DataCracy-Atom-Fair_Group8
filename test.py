from numpy import histogram
from numpy.lib.function_base import append
import streamlit as st
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials #-> Để nhập Google Spreadsheet Credentials
import gspread
from google.oauth2 import service_account
import datetime
from gspread_dataframe import set_with_dataframe
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive'],
)
gc1 = gspread.authorize(credentials)
spreadsheet_key = '1lAlQLlUBaqCL1Onh0JfK9pZGEbzqSCupCPo1r3UGj6c'


today = datetime.date.today()
# today
sh1=gc1.open("Kho mẫu - Test").worksheet('Sheet1')
sample_name=sh1.get_all_records()
sample_name_pd=pd.DataFrame(sample_name)
step=st.selectbox('Chọn thao tác',['Trả mẫu','Mượn mẫu'])
factory=st.selectbox('Chọn bộ phận',['NM1','NM3','X4','TD','NM NỆM','QLCL','THU MUA','P.TM'])
filter=st.multiselect('Chọn Khách hàng',sample_name_pd['TÊN KHÁCH HÀNG'].unique().tolist())
sample=sample_name_pd[sample_name_pd['TÊN KHÁCH HÀNG'].isin(filter)]

sp=st.multiselect('Chọn sản phẩm',sample['TÊN SẢN PHẨM'].unique())
table=sample[sample['TÊN SẢN PHẨM'].isin(sp)]
table_df=table[['TÊN KHÁCH HÀNG','TÊN CHI TIẾT']].reset_index(drop=True)
table_df['NGÀY'],table_df['THAO TÁC'],table_df['BỘ PHẬN']=today,step,factory
table_df
# sub_folders=order_df[['TÊN KHÁCH HÀNG','TÊN SẢN PHẨM']]
# sub_folders.set_index('TÊN KHÁCH HÀNG').T.to_dict('list')

if st.button('Xuất danh sách'):
# pdf=table_df.to_pd
    dict_id={}
    sheet_index_no1= 1

    spreadsheet_key = '1lAlQLlUBaqCL1Onh0JfK9pZGEbzqSCupCPo1r3UGj6c' # input SPREADSHEET_KEY HERE
    sh = gc1.open_by_key(spreadsheet_key)
    worksheet1 = sh.get_worksheet(sheet_index_no1)#-> 0 - first sheet, 1 - second sheet etc. 

    import gspread_dataframe as gd
    import gspread as gs

    ws = gc1.open("Kho mẫu - Test").worksheet('Sheet2')
    existing = gd.get_as_dataframe(ws)
    updated = existing.append(table_df)
    gd.set_with_dataframe(ws, updated)




# # APPEND DATA TO SHEETimport gspread_dataframe as gd
# import gspread_dataframe as gd
# import gspread as gs
# # gc = gs.service_account(filename="your/cred/file.json")

# def export_to_sheets(worksheet_name,df,mode='r'):
#     ws = gc1.open("Kho mẫu - Test").worksheet(worksheet_name)
#     if(mode=='w'):
#         ws.clear()
#         gd.set_with_dataframe(worksheet=ws,dataframe=df,include_index=False,include_column_header=True,resize=True)
#         return True
#     elif(mode=='a'):
#         ws.add_rows(df.shape[0])
#         gd.set_with_dataframe(worksheet=ws,dataframe=df,include_index=False,include_column_header=False,row=ws.row_count+1,resize=False)
#         return True
#     else:
#         return gd.get_as_dataframe(worksheet=ws)
    
# # df = pd.DataFrame.from_records([{'a': i, 'b': i * 2} for i in range(100)])
# export_to_sheets("Sheet2",table_df,'a')
