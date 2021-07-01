import streamlit as st
import smtplib
import re #-> Để xử lý data dạng string
from datetime import datetime as dt #-> Để xử lý data dạng datetime
import base64
from io import BytesIO
import numpy as np
import pandas as pd #-> Để update data dạng bản
pd.plotting.register_matplotlib_converters()
import matplotlib.image as mpimg
from google.oauth2 import service_account
from gsheetsdb import connect
from datetime import datetime, timedelta
from datetime import datetime as dt
from typing import Text
from numpy.core.numeric import NaN
import logging.handlers
from streamlit.elements import empty
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
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

def check_student(df,id):
    id_column=st.sidebar.selectbox('Chosse index:',
                        df.columns.tolist())
    df[id_column]=df[id_column].astype(str)
    st_id=df.loc[df[id_column]==id]
    st_id=st_id.set_index(id_column)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = st_id.select_dtypes(include=numerics)
    st.markdown('### Hello {}'.format(id))
    marks=numerical_cols
    marks["Student_ID"]=id
    mark=marks.melt(id_vars='Student_ID',var_name='Object',value_name='Scores')
    sns.set(style='darkgrid', font_scale=2, rc={"figure.figsize": [14, 6]})
    f, ax = plt.subplots(1, 1, figsize=(25, 10))
    g = sns.lineplot(x='Object',y='Scores',data=mark, ax=ax)
    ax.set_title('histogram of marks about student')
    ax.set_ylabel('Marks')
    ax.set_xlabel('Object')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'



def transform(df,email_list):
  # SUMMARY
    index=st.sidebar.multiselect('Chosse index:',
                        df.columns.tolist())
    index1=index[0]                 
    index2=index[1]
    index3=index[2]
    df=df.set_index([index1,index2,index3])

    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    category=['object','bool']
    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    _pass=st.sidebar.number_input('Enter mark to pass exam:', step=1)
    numerical_cols = df.select_dtypes(include=numerics)
    category_cols=  df.select_dtypes(include=category)
    email_list=email_list.set_index(index1)
    first_cols=st.sidebar.multiselect('First academic Object names',
                        numerical_cols.columns.tolist(),
                        numerical_cols.columns.tolist())
    second_cols=st.sidebar.multiselect('Second academic Object names',
                        numerical_cols.columns.tolist(),
                        numerical_cols.columns.tolist())

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
    st.set_option('deprecation.showPyplotGlobalUse', False)
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

    st.markdown('### SỐ LƯỢNG SINH VIÊN ĐÃ HỌC XONG CÁC MÔN NỀN TẢNG')
    fig, ax = plt.subplots()
    sns.barplot(data=first_pass,x=first_pass[index2],y="Scores")
    st.pyplot()
    st.markdown("")
    st.markdown("### SỐ LƯỢNG SINH VIÊN ĐÃ HỌC XONG CÁC MÔN BẮT BUỘC")
    if len(_pass18)==0:
      st.write("No one student in here")
    else:
      st.set_option('deprecation.showPyplotGlobalUse', False)
      sns.barplot(data=_pass18,x=_pass18[index2],y="Scores")
      st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown('RATIO')
    a.plot(x='Intake', y=['Terminate','Doing'], kind="bar")
    st.pyplot()
    #warning list 1:
    year=st.sidebar.slider("Years", min_value=2014, max_value=2021, step=1)
    fail_O1=first_fail.loc[(first_fail.Object=='OMAT')|(first_fail.Object=='OSTA')]
    a=fail_O1[index1]
    a=a.reset_index()
    

    fail_O2=first_fail.loc[(first_fail.Object!='OMAT')|(first_fail.Object!='OSTA')]
    Id_count=fail_O2.groupby(fail_O2[index1]).Scores.count()
    list2=Id_count.loc[Id_count.values>=4]
    b=pd.DataFrame(list2.index)
    w_list1=pd.concat([a,b])
    wlist1=pd.DataFrame(w_list1[index1])
    wl1=wlist1.merge(df,how='left',on=index1)
    warning_list1=wl1
    #Warning list 2:
    sub=['2015','2016','2017','2018']
    for i in sub:
      list2=first_fail.loc[first_fail[index2].str.contains(i)]
    
    c=list2[index1] #.tolist()
    wlist2=pd.DataFrame(c)
    wl2=wlist2.merge(df,how='left',on=index1)
    warning_list2=wl2
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
    list_BSEM=BSEM_list#.index.tolist()   
    cho=st.selectbox('Chosse list',['Warning list 1','Warning list 2','BSEM list'])
    if cho=='Warning list 1':
      file=warning_list1
    elif cho=='Warning list 2':
      file=warning_list2
    else:
      file=list_BSEM
    file
    if st.button('Download here'):
      tmp_download_link = download_link(file, 'YOUR_DF.csv', 'Click here to download your data!')
      st.markdown(tmp_download_link, unsafe_allow_html=True)
       
def main():
    st.title('Student a dataset')
    files = st.file_uploader("Upload file", type=['csv','xlsx','pickle'],accept_multiple_files=True)
    if not files:
        st.write("Upload a .csv or .xlsx file to get started")
    else:
      df = get_df(files[0])
      email_list=get_df(files[1])
      email_list=email_list.replace((0,np.nan))
      data_df=df.replace((0,np.nan))
      data_df.columns=data_df.columns.str.replace(" ","_")
      choose=st.sidebar.selectbox('Enter your choose:',['Operation Dashboard','Student checking'])
      if choose=='Operation Dashboard':
          transform(df,email_list)
      else:
          st_id=st.sidebar.text_input('Enter Student ID:',"")
          check_student(df,st_id)

main()


