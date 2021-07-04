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
from typing import Sized
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
# import pandas_profiling as pp
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans

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
@st.cache
def check_student(df,id):

    id_column=st.sidebar.multiselect('Chọn trường chứa mã sinh viên và họ tên:',
                        df.columns.tolist())
    df[id_column]=df[id_column].astype('str')
    id_st=id_column[0]
    name=id_column[1]
    if  df[id_st].str.contains(id).any()==False:
      st.error('Mã số sinh viên chưa đúng. Vui lòng kiểm tra lại')
    else:
      st_id=df.loc[df[id_st]==id]
      st_id=st_id.set_index(id_st)
      numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
      numerical_cols = st_id.select_dtypes(include=numerics)
      st_name=st_id[name].values.tolist()
      st.markdown('### Hello {}'.format(st_name))
      marks=numerical_cols
      marks["Student_ID"]=id
      mark=marks.melt(id_vars='Student_ID',var_name='Object',value_name='Scores')
      mark=mark.loc[mark.Scores!=0]
      sns.set(style='darkgrid', font_scale=2, rc={"figure.figsize": [20,10]})
      # f, ax = plt.subplots(1, 1, figsize=(30, 15))
      # a=sns.lineplot((x=avg,y=
      x=mark['Object']
      y=mark['Scores']
            # Create some random data
      y_mean = [np.mean(y)]*len(x)

      fig,ax = plt.subplots()
      data_line = ax.plot(x,y, label='Điểm số', marker='o')
      mean_line = ax.plot(x,y_mean, label='Trung bình', linestyle='--')
      # Make a legend
      st.markdown('#### PHỔ ĐIỂM THEO MÔN HỌC')
      legend = ax.legend(loc='upper right')
      ax.set_ylabel('ĐIỂM SỐ')
      ax.set_xlabel('MÔN HỌC')
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def clustering(df):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics)
    index=st.sidebar.selectbox('CHỌN TRƯỜNG THÔNG TIN ĐỂ PHẦN NHÓM',
                            df.columns.tolist())     
    if not index:
      st.error('Please enter variable at sidebar')       
    else:  
      df=df.set_index([index]) 
      obj=st.sidebar.multiselect('CHỌN MÔN HỌC',
                              numerical_cols.columns.tolist(),
                              numerical_cols.columns.tolist())
      if not obj:
        st.sidebar.error('Please enter variable at sidebar')       
      else:
        n_clus=st.sidebar.number_input('CHỌN SỐ LƯỢNG NHÓM',step=1)
        if not n_clus:
          st.sidebar.error('Please enter variable at sidebar')       
        else:
          perc =[.25, .50, .75,.90]
        # list of dtypes to include
          include =[ 'float', 'int']
          kmeans2 = KMeans(n_clusters=n_clus) #number of cluster = 4
          list=[]
          for i in obj:
            list.append(i)
          y = df.loc[:,list]
          Y =y.reset_index()
    
          desc =y.describe(percentiles = perc)
          desc=desc.transpose()
          st.markdown('PHÂN TÍCH TỔNG QUAN')
          desc
          Y["cluster"] = kmeans2.fit_predict(Y)
          st.markdown('PHÂN NHÓM SINH VIÊN THEO PHƯƠNG PHÁP CLUSTERING')
          agg = Y.groupby('cluster')[obj].mean().reset_index()
          fin= agg
          fin
      #   #Print student ID based on clustering 
          for i in range(n_clus): 

              df_tmp0 = Y.loc[Y.cluster == i] #Cluster level from 0 to 3

              st.markdown('Danh sách sinh viên thuộc nhóm {}'.format(i+1))
              student_id0=df_tmp0
              df_tmp0
              tmp_download_link = download_link(df_tmp0, 'YOUR_DF.csv', 'Click here to download your data!')
              st.markdown(tmp_download_link, unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def transform(df):

  # SUMMARY
    st.sidebar.markdown('A. XÁC ĐỊNH TRƯỜNG THÔNG TIN')
    index=st.sidebar.multiselect('Chọn thông tin cần xem báo cáo:',
                        df.columns.tolist())
    index1=0
    index2=0
    index3=0
    if not index:
      st.error('Vui lòng chọn trường thông tin tại sidebar')
    else:
      index1=index[0]                 
      index2=index[1]
      index3=index[2]
      df=df.set_index([index1,index2,index3])
    
    
      numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
      category=['object','bool']
      df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
      _pass=st.sidebar.number_input('Mức điểm qua môn:', step=1)
      st.sidebar.markdown('B. ĐIỀU KIỆN ĐỂ TÌM DANH SÁCH SINH VIÊN')
      numerical_cols = df.select_dtypes(include=numerics)
      category_cols=  df.select_dtypes(include=category)
      first_cols=st.sidebar.multiselect('Các môn bắt buộc nhóm 1 / môn nền tảng năm nhất',
                          numerical_cols.columns.tolist())
      second_cols=st.sidebar.multiselect('Các môn bắt buộc nhóm 2 / môn nền tảng năm 2',
                          numerical_cols.columns.tolist())
      num=st.sidebar.number_input('Tổng số môn học nền tảng bắt buộc:',step=1)
      # major=(category_cols.reset_index().columns.tolist())
      # # -index
      # m=[m for m in major if m not in index]
      # list_major=category_colsunique()
      # list_major

      number=numerical_cols.reset_index()
      ds_df=number.reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
      first_df=df[first_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')

      second_df=df[second_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
      first_pass=first_df.loc[first_df.Scores>=_pass]
      first_fail=first_df.loc[(first_df.Scores<_pass)]
      ds_pass=ds_df.loc[ds_df.Scores>=_pass]
      l=ds_pass.groupby([ds_pass[index1],ds_pass[index2]]).count()
      l=l.reset_index()
      _pass18=l.loc[l.Scores>=num]
      pass18=_pass18.groupby(_pass18[index2]).count()
      pass18= pass18.reset_index()
      ds_terminate=ds_df.loc[(ds_df[index3].isnull()==False)] 
      terminate=ds_terminate.groupby(ds_terminate[index2]).count()
      ds_doing=ds_df.loc[(ds_df[index3].isnull()==True)]
      doing=ds_doing.groupby(ds_doing[index2]).count()
      doing['Đã thôi học']=terminate['Scores']
      doing['Đang theo học']=doing['Scores']
      doing['Phần trăm']=(doing['Đã thôi học']/doing['Đang theo học'])*100
      a=doing[['Đang theo học','Đã thôi học','Phần trăm']]
      a=a.reset_index()
      st.title('A. BÁO CÁO TỔNG QUAN TÌNH HÌNH LỚP HỌC')
      st.markdown("#### 1. PHỔ ĐIỂM TRUNG BÌNH CỦA NHÓM 1")

      # st.set_option('deprecation.showPyplotGlobalUse', False)
      # hu=sns.FacetGrid(first_df,row=index2,col='Object')
      # hu.map(sns.histplot,'Scores')
      # sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [60,20]})
    

      a=df[first_cols]
      ii=a.columns.tolist()
      x=round(len(ii)/2)
      y=2
      c = 1  # initialize plot counter
      fig = plt.figure(figsize=(20,15))
      for i in ii:
          plt.subplot(x, y, c)
          plt.title('{}, subplot: {}{}{}'.format(i, x, y, c))
          plt.xlabel(i)
          sns.countplot(df[i])
          c = c + 1

      plt.show()
      st.pyplot(fig)
      st.markdown("#### 2. PHỔ ĐIỂM TRUNG BÌNH CỦA NHÓM 2")
      h=df[second_cols]
      iii=h.columns.tolist()
      d=round(len(iii)/2)
      e=2
      f= 1  # initialize plot counter
      fig2 = plt.figure(figsize=(25,30  ))
      for z in iii:
          plt.subplot(d, e, f)
          plt.title('{}, subplot: {}{}{}'.format(z, d, e, f))
          plt.xlabel(z)
          sns.countplot(df[z])
          f = f + 1
      plt.show()
      st.pyplot(fig2)

      st.markdown('### 3. SỐ LƯỢNG SINH VIÊN ĐÃ HỌC XONG CÁC MÔN NỀN TẢNG (NHÓM 1 & NHÓM 2')
      fig, ax = plt.subplots()
      sns.barplot(data=first_pass,x=first_pass[index2],y="Scores")
      st.pyplot()
      st.markdown("")
      st.markdown("### 4. SỐ LƯỢNG SINH VIÊN ĐÃ HỌC XONG CÁC MÔN BẮT BUỘC")
      if len(pass18)==0:
        st.write("Chưa có sinh viên nào hoàn thành tất cả các môn học")
      else:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.barplot(data=pass18,x=index2,y=index1)
        st.pyplot()
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.markdown('### 5. TỈ LỆ SINH VIÊN ĐANG THEO HỌC VÀ ĐÃ NGHỈ HỌC')
      a.plot(x=index2, y=['Đã thôi học','Đang theo học'], kind="bar")
      st.pyplot()
      #warning list 1
      
      st.sidebar.markdown('C. TÌM SINH VIÊN RỚT NĂM 1')
      need=st.sidebar.multiselect('Đậu các môn bắt buộc',                      
                          numerical_cols.columns.tolist())
      total=st.sidebar.number_input('Hoặc đạt tổng số môn cần đạt:',step=1)
      for i in need:

        fail_O1=first_fail.loc[(first_fail.Object==i)]
        fail_O2=first_fail.loc[(first_fail.Object!=i)]

      a=fail_O1[index1]
      a=a.reset_index()
      Id_count=fail_O2.groupby(fail_O2[index1]).Scores.count()
      list2=Id_count.loc[Id_count.values >=total]
      b=pd.DataFrame(list2.index)
      w_list1=pd.concat([a,b])
      wlist1=pd.DataFrame(w_list1[index1])
      wl1=wlist1.merge(df,how='left',on=index1)
      warning_list1=wl1
      st.sidebar.markdown('D. TÌM SINH VIÊN BỊ BUỘC THÔI HỌC')
      year=st.sidebar.multiselect('Sinh viên thuộc nhóm rớt môn năm nhất và thuộc các niêm khóa:',['2014','2015','2016','2017','2018','2019','2020','2021'])

      #Warning list 2:
      for i in year:
        list2=first_fail.loc[first_fail[index2].str.contains(i)]
        list2
      c=list2[index1] #.tolist()
      wlist2=pd.DataFrame(c)
      wl2=wlist2.merge(df,how='left',on=index1)
      warning_list2=wl2
      #List pass B
      _passB=ds_pass.loc[ds_pass.Object.str.contains('B')]
      count_passB=_passB.groupby([ds_pass[index1],ds_pass.Object]).count()
    

      #list study BSEM
      list_BSEM=[]
      BSEM=ds_pass.groupby(ds_pass[index1]).Object.count()
      BSEM_list=BSEM.loc[BSEM.values==18]
      list_BSEM=BSEM_list#.index.tolist()   
      st.markdown('### 6. DANH SÁCH SINH VIÊN THUỘC NHÓM CẢNH CÁO')
      cho=st.selectbox('Vui lòng chọn',['DANH SÁCH SINH VIÊN CẦN ĐƯỢC CẢNH BÁO','DANH SÁCH SINH VIÊN BUỘC THÔI HỌC','BSEM list'])

      if cho=='DANH SÁCH SINH VIÊN CẦN ĐƯỢC CẢNH BÁO':
        file=warning_list1
      elif cho=='DANH SÁCH SINH VIÊN BUỘC THÔI HỌC':
        file=warning_list2
      else:
        file=list_BSEM
      file
      if st.button('Tải Danh sách tại đây'):
        tmp_download_link = download_link(file, 'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

      
def main():


    st.markdown("<p style='text-align: center;'><strong><span style='font-size: 28px; font-family: Arial, Helvetica, sans-serif;color:orange'>ỨNG DỤNG</span></strong></p><p style='text-align: center;'><span style='font-family: Arial, Helvetica, sans-serif;'><span style='font-size: 28px;color: orange'><strong>HỖ TRỢ QUẢN L&Yacute; TRONG GI&Aacute;O DỤC</strong></span></span></p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right;'><strong><em><span style='font-size: 12px;'>Tạo bởi: Atom Fair - Nhóm 8 (</span></em></strong><span style='font-family: Arial, Helvetica, sans-serif;'><span style='font-size: 12px;'><em><strong>Lâm Hiếu -&nbsp;</strong></em></span></span><span style='font-family: Arial, Helvetica, sans-serif;'><span style='font-size: 12px;'><em><strong>Toàn Trần;</strong></em></span></span><strong><em><span style='font-size: 12px; font-family: Arial, Helvetica, sans-serif;'>- Hạnh Nguyễn)</span></em></strong></p>", unsafe_allow_html=True)
    files = st.file_uploader("Upload file", type=['csv','xlsx','pickle'],accept_multiple_files=False)
    
    if not files:
          st.write("Upload a .csv or .xlsx file to get started")
    else:
      df = get_df(files)
      data_df=df.replace((0,np.nan))
      data_df.columns=data_df.columns.str.replace(" ","_")
      choose=st.sidebar.selectbox('Enter your choose:',['Operation Dashboard','Student checking','Phân nhóm học tập'])
      if choose=='Operation Dashboard':
          transform(df)
      elif choose=='Student checking':
          st_id=st.sidebar.text_input('Nhập mã sinh viên:',"")
          if not st_id:
            st.sidebar.error('Vui lòng nhập mã số sinh viên')
          else:
           check_student(df,st_id)
      elif choose == 'Phân nhóm học tập':
          clustering(df)


main()


