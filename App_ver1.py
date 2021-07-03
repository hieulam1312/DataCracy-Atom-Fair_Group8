import streamlit as st
import smtplib
from PIL import Image
import re #-> Để xử lý data dạng string
from datetime import datetime as dt #-> Để xử lý data dạng datetime
import gspread #-> Để update data lên Google Spreadsheet
from gspread_dataframe import set_with_dataframe #-> Để update data lên Google Spreadsheet
import numpy as np
import pandas as pd #-> Để update data dạng bản
import json 

import matplotlib.image as mpimg
from google.oauth2 import service_account
from gsheetsdb import connect
from datetime import datetime, timedelta
from datetime import datetime as dt
from typing import Text
from numpy.core.numeric import NaN
import streamlit as st
import json
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt


# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive'],
)
gc1 = gspread.authorize(credentials)
spreadsheet_key = '1_APNLrt6uNo6aaZIERHYAssD_KEU2pEqEnInG1ewnWY' # input SPREADSHEET_KEY HERE

gc2=gspread.authorize(credentials)
spreadsheet_key='1Kf79UeBTa0q2NAh4PaW2Y1nqE__S0wiSQSOkk2dkQm0'
#email_list
sh1=gc1.open("Group 8 - Atom Fair - Dataset").worksheet('Email list')
data1=sh1.get_all_records()
email_list=pd.DataFrame(data1)
email_list=email_list.replace("",np.nan)
email_list.columns=email_list.columns.str.replace(" ","_")
email_list['Student_ID'].astype(str)

#td_df
sh2=gc1.open("Group 8 - Atom Fair - Dataset").worksheet('Results')
data=sh2.get_all_records()
data_df=pd.DataFrame(data)
data_df=data_df.replace((0,np.nan))
data_df.columns=data_df.columns.str.replace(" ","_")

#EDA - DASHBOARD OPERATION

data_df['Intake'].value_counts().plot.pie()


#B. SEND MAIL BOT
ds_df=data_df.melt(id_vars=['Student_ID','Intake','NOTE'],var_name='Object',value_name='Scores')

#List of subject
list_O=['OBRW','OVWL','OFIN','OMAT','OSTA','OWIN','OREC','OMAR']
list_B=['BFIN','BMIK','BMGT','BACC','BWET','BMAK']
list_P_BA=['PMAR','PWIN']
list_P_FA=['PFIN','PACC']

#Warning_list1: 
_pass=50
warning_list1=[]
fail=ds_df.loc[(ds_df.Scores<50)]
fail_O=fail.loc[(fail.Object.str.contains('O'))]
_pass=ds_df.loc[ds_df.Scores>=50]
fail_O1=fail.loc[(fail.Object=='OMAT')|(fail.Object=='OSTA')]
a=fail_O1.Student_ID.values.tolist()
fail_O2=fail.loc[(fail.Object.str.contains('O'))&((fail.Object!='OMAT')|(fail.Object!='OSTA'))]
Id_count=fail_O2.groupby('Student_ID').Scores.count()
list2=Id_count.loc[Id_count.values>=4]
b=list2.index.tolist()
warning_list1=a+b
#Warning list 2:
sub=['2015','2016','2017','2018']
for i in sub:
    list2=fail_O.loc[fail_O.Intake.str.contains(i)]
warning_list2=list2.Student_ID.tolist()


#List pass B
_passB=_pass.loc[_pass.Object.str.contains('B')]
count_passB=_passB.groupby(['Student_ID','Object']).count()
count_passB=count_passB.reset_index()
_2=['BFIN','BACC','BMGT']
study_P_list=[]
for i in _2:
    list_B=count_passB.loc[count_passB.Object==i]
    count_pass_B=list_B.groupby('Student_ID').count()
list3=count_pass_B.loc[count_pass_B.Object>=3]
study_P_list.append(list3.index.tolist())

#list study BSEM
list_BSEM=[]
BSEM=_pass.groupby('Student_ID').Object.count()
BSEM_list=BSEM.loc[BSEM.values==18]
list_BSEM=BSEM_list.index.tolist()



#EDA - DASHBOARD OPERATION


st.markdown("<h1 style='text-align: center; color: blue;font-style:bold'>PEFORMANCE STUDY</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: right; color:black;font-style: italic'> Created by Group 8 - Atom Fair</h4>", unsafe_allow_html=True)
st.markdown("")
#Count of class

#Số lượng pass O và 18 môn
O=_pass.loc[_pass.Object.str.startswith('O')].groupby('Intake').count()
_passO=O.reset_index()
l=_pass.groupby(['Intake','Student_ID']).count()
l=l.reset_index()
_pass18=l.loc[l.Scores>=18]
st.markdown('### SỐ LƯỢNG SINH VIÊN ĐẬU MÔN O THEO LỚP')
fig, ax = plt.subplots()
sns.barplot(data=_passO,x="Intake",y="Scores")
st.pyplot(fig)
st.markdown("")
st.markdown("### SỐ LƯỢNG SINH VIÊN ĐÃ HỌC XONG 18 MÔN O,B,P")
sns.barplot(data=_pass18,x="Intake",y="Scores")
st.pyplot(fig)
#Phân bố điểm của lớp

st.sidebar.selectbox('Enter Student ID:',data_df['Student_ID'])

































#CREATED SEND MAIL BOT
Take_action=['Warning for first academic year','Warning for second academic year']


def main():
    
    email_sender=st.text_input('Enter User Email: ')
    password=st.text_input('Enter User password: ',type='password')

    for i in Take_action:
        st.button(i)
        if i=='Warning for first academic year':  
            for y in warning_list1:
                email=email_list.loc[email_list.Student_ID==y]

        elif i=='Warning for second academic year':
            for z in warning_list2:
                email=email_list.loc[email_list.Student_ID==z]
    
    email_reciever=email['Email'].to_string(index=False)
    subject=st.text_input('Subject: ')
    body=st.text_area('Context')
    if st.button("Send Email"):
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(email_sender, password) #login with mail_id and password
        messages="Subject: {}\n\n{}".format(subject,body)
        session.sendmail(email_sender, email_reciever,messages)
        session.quit()

main()

