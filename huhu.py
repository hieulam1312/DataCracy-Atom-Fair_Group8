import streamlit as st

import re #-> Để xử lý data dạng string
from datetime import datetime as dt #-> Để xử lý data dạng datetime

import numpy as np
import pandas as pd #-> Để update data dạng bản

import matplotlib.image as mpimg
from google.oauth2 import service_account
from gsheetsdb import connect
from datetime import datetime, timedelta
from datetime import datetime as dt
from typing import Text
from numpy.core.numeric import NaN
import streamlit as st

from streamlit.elements import empty
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def get_df(file):
  # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(file)
  return df.replace((" ",np.nan))

def transform(df):
  # SUMMARY
    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                   'bool'])].index.values
    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                   'bool'])].index.values
    category_cols=df_types[~df_types['Data Type'].isin(['object',
                   'bool'])].index.values
    index=st.sidebar.multiselect('Chosse index:',
                        df.columns.tolist())
    index1=index[0]
    index2=index[1]
    index3=index[2]
    df=df.set_index([index1,index2,index3])
    first_cols=st.sidebar.multiselect('First academic Object names',
                        numerical_cols.tolist(),
                        numerical_cols.tolist())
    second_cols=st.sidebar.multiselect('Second academic Object names',
                        df.columns.tolist(),
                        df.columns.tolist())
    _pass=st.sidebar.number_input('Enter mark to pass exam:', step=1)
    ds_df=df.reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
    first_df=df[first_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
    second_df=df[second_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
    first_pass=first_df.loc[first_df.Scores>=_pass]
    first_fail=first_df.loc[(first_df.Scores<_pass)]
    ds_pass=ds_df.loc[ds_df.Scores>=_pass]
    l=ds_pass.groupby([ds_pass[index1],ds_pass[index2]]).count()
    l=l.reset_index()
    _pass18=l.loc[l.Scores>=18]
    ds_terminate=ds_df.loc[(ds_df[index3].isnull()==False)] 
    terminate=ds_terminate.groupby(ds_terminate[index2]).count()
    ds_doing=ds_df.loc[(ds_df[index3].isnull()==True)]
    doing=ds_doing.groupby(ds_doing[index2]).count()
    doing['Terminate']=terminate['Scores']
    doing['Doing']=doing['Scores']
    doing['Ratio']=(doing['Terminate']/doing['Doing'])*100
    a=doing[['Doing','Terminate','Ratio']]
    a=a.reset_index()
    st.title('DATASET OF MARKS')
    st.markdown("histogram of marks")
    sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [14, 6]})
    f, ax = plt.subplots(1, 1, figsize=(14, 6))
    g = sns.histplot(x=first_df.Scores, data=first_df, ax=ax)
    ax.set_title('histogram of marks about first academy')
    ax.set_ylabel('Students')
    ax.set_xlabel('Marks')
    st.pyplot()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [14, 6]})
    f, ax = plt.subplots(1, 1, figsize=(14, 6))
    g = sns.histplot(x=second_df.Scores, data=second_df, ax=ax)
    ax.set_title('histogram of marks about second academy')
    ax.set_ylabel('Students')
    ax.set_xlabel('Marks')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.markdown('### SỐ LƯỢNG SINH VIÊN ĐẬU MÔN O THEO LỚP')
    fig, ax = plt.subplots()
    sns.barplot(data=first_pass,x=first_pass[index2],y="Scores")
    st.pyplot()
    st.markdown("")
    st.markdown("### SỐ LƯỢNG SINH VIÊN ĐÃ HỌC XONG 18 MÔN O,B,P")
    if len(_pass18)==0:
      st.write("No one student in here")
    else:
      sns.barplot(data=_pass18,x=_pass18[index2],y="Scores")
      st.pyplot()

    st.markdown('RATIO')
    a.plot(x='Intake', y=['Terminate','Doing'], kind="bar")
    st.pyplot()
#Phân bố điểm của lớp






    #warning list 1:
    year=st.sidebar.slider("Years", min_value=2014, max_value=2021, step=1)
    warning_list1=[]

    fail_O1=first_fail.loc[(first_fail.Object=='OMAT')|(first_fail.Object=='OSTA')]
    a=fail_O1[index1].values.tolist()

    fail_O2=first_fail.loc[(first_fail.Object!='OMAT')|(first_fail.Object!='OSTA')]
    Id_count=fail_O2.groupby(fail_O2[index1]).Scores.count()
    list2=Id_count.loc[Id_count.values>=4]
    b=list2.index.tolist()
    warning_list1=a+b

    #Warning list 2:
    sub=['2015','2016','2017','2018']
    for i in sub:
        list2=first_fail.loc[first_fail[index2].str.contains(i)]
    warning_list2=list2[index1].tolist()
    
    #List pass B
    _passB=ds_pass.loc[ds_pass.Object.str.contains('B')]
    count_passB=_passB.groupby([ds_pass[index1],ds_pass.Object]).count()
    # count_passB=count_passB.reset_index()
    _2=['BFIN','BACC','BMGT']
    study_P_list=[]
    for i in _2:
        list_B=count_passB.loc[count_passB.Object==i]
        count_pass_B=list_B.groupby(list_B[index1]).count()
    list3=count_pass_B.loc[count_pass_B.Object>=3]
    study_P_list.append(list3.index.tolist())

    #list study BSEM
    list_BSEM=[]
    BSEM=ds_pass.groupby(ds_pass[index1]).Object.count()
    BSEM_list=BSEM.loc[BSEM.values==18]
    list_BSEM=BSEM_list.index.tolist()



def main():
    st.title('Student a dataset')
    file1 = st.file_uploader("Upload file", type=['csv','xlsx','pickle'])
    if not file1:
        st.write("Upload a .csv or .xlsx file to get started")
    else:
        df = get_df(file1)
        data_df=df.replace((0,np.nan))
        data_df.columns=data_df.columns.str.replace(" ","_")
        if st.sidebar.button('Operation Dashboard'):
          transform(df)
        else:
          st.write("")
    return transform(df)  

main()


