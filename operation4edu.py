from enum import unique
from inspect import FrameInfo
from types import new_class
import matplotlib
import streamlit as st
import smtplib
import re #-> Để xử lý data dạng string
from datetime import datetime as dt #-> Để xử lý data dạng datetime
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from streamlit.proto.RootContainer_pb2 import SIDEBAR #-> Để update data dạng bản
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
  return df

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
      st.markdown('### Hello {}'.format(st_name[0]))
      marks=numerical_cols
      marks["Student_ID"]=id
      mark=marks.melt(id_vars='Student_ID',var_name='Object',value_name='Scores')
      mark=mark.loc[mark.Scores!=0]
      sns.set(style='whitegrid', font_scale=4, rc={"figure.figsize": [35,25]})
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
    sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics)
    index=st.sidebar.selectbox('CHỌN TRƯỜNG CHỨA MÃ SỐ SINH VIÊN',
                            df.columns.tolist())     
    if not index:
      st.error('Vui lòng bổ sung trường thông tin tại sidebar')       
    else:  
      df=df.set_index([index]) 
      obj=st.sidebar.multiselect('CHỌN MÔN HỌC',
                              numerical_cols.columns.tolist())
      
      kmeans2 = KMeans(n_clusters=4) #number of cluster = 4
      _list=[]
      for i in obj:
        _list.append(i)
      y = df.loc[:,_list]
      Y =y 
      St_ID=y.reset_index()
      Y["cluster"] = kmeans2.fit_predict(Y)
      Y["cluster"] = Y["cluster"].astype("category")
      Y["cluster"] = kmeans2.labels_
      st.markdown("")
      st.markdown('PHỔ ĐIỂM TRUNG BÌNH')
      mean_df = df.iloc[:,3:25] #Create a temporary df to calculate mean values
  # print(mean_df)
      row = df.iloc[0,3:25] # clus_df.iloc[clus_df["Student ID"] = x,3:25]
      # Plot a chart with selected row, can be replaced value 0 with input student ID
      values = list(row) #create a list contains grades
      plt.figure(figsize = (25,10))
      ax = row.plot(kind='bar', label='Grade')
      mean_df.mean().plot(ax=ax, color='r', linestyle='-', label='Mean')
      ax.legend()
      st.set_option('deprecation.showPyplotGlobalUse', False)
      plt.xlabel("Grade per subject")
      plt.show()
      st.pyplot()
      # Silhouette Coefficient to find optimal cluster
      from sklearn.metrics import silhouette_score
      silhouette_coefficients = []
      kmeans_kwargs = {
          "init": "random",
          "n_init": 10,
          "max_iter": 300,
          "random_state": 42,
      }
      clus_dict = {} #contain number of cluster and silhouette coef
      #Start at 2 clusters
      for k in range(2, 11):
          kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
          kmeans.fit(Y)
          score = silhouette_score(Y, kmeans.labels_)
          silhouette_coefficients.append(score)
          clus_dict[k] = score
      st.markdown('TÌM SỐ NHÓM TỐI ƯU THEO PHƯƠNG PHÁP SILHOUETTE')
      #Draw chart to visualize clusters
      fig=plt.figure(figsize = (25,15))
      plt.plot(range(2,11), silhouette_coefficients)
      plt.xticks(range(2,11))
      plt.xlabel("Number of clusters")
      plt.ylabel("Silhouette Coefficient")
      plt.show()
      st.pyplot(fig)
      #Print and draw chart with optimal number of clusters
      max_value = max(clus_dict, key=clus_dict.get)
      st.markdown('Số nhóm tối ưu có thể chia là: ' + str(max_value))
      st.markdown("")
      number_clus=st.number_input('Nhập số nhóm muốn cluster tại đây:',step=1)
      st.markdown('KẾT QUẢ PHÂN NHÓM VỚI CỤM TỐI ƯU')
      st.markdown("")
      if not number_clus:
        kmeans = KMeans(n_clusters=max_value) #number of cluster = max value
        n_clus=max_value
      else:
        kmeans = KMeans(n_clusters=number_clus) #number of cluster = max value
        n_clus=number_clus
      Y["cluster"] = kmeans.fit_predict(Y)
      Y["cluster"] = Y["cluster"].astype("category")
      print(Y)
      kmeans.fit(Y)
      Y["cluster"] = kmeans.labels_
      sb.pairplot(data=Y,hue="cluster",palette='viridis',height=4)
      st.pyplot()
      
      Y["cluster"] = kmeans2.fit_predict(Y)
      if st.button('LẤY DANH SÁCH KẾT QUẢ'):
    #   #Print student ID based on clustering 
        for i in range(n_clus): 
            df_tmp0 = Y.loc[Y.cluster == i] #Cluster level from 0 to 3
            st.markdown('Danh sách sinh viên thuộc nhóm {}'.format(i+1))
            df_tmp0=df_tmp0.reset_index()
            df_tmp0
            tmp_download_link = download_link(df_tmp0, 'YOUR_DF.csv', 'Bấm vào đây để tải file!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
def transform(df,first,index2,second):

      st.write('### A. BÁO CÁO TỔNG QUAN TÌNH HÌNH LỚP HỌC')
      st.markdown("#### 1. PHỔ ĐIỂM TRUNG BÌNH THEO MÔN HỌC")
      a=df[first]
      ii=a.columns.tolist()
      x=round(len(ii)/2)
      y=2
      c = 1  # initialize plot counter
      fig = plt.figure(figsize=(25,15))
      for i in ii:
          plt.subplot(x, y, c)
          plt.title('{}'.format(i))
          plt.xlabel(i)
    
          sns.countplot(df[i],color='Green')
          c = c + 1
      st.pyplot(fig)
      st.markdown('**Ghi chú:**')
      st.markdown('- Trục X: Các mức điểm mà sinh viên đạt được theo môn học')
      st.markdown('- Trục Y: Số lượng học viên đạt được')
      st.markdown('- Có thể tùy chọn môn, thêm hoặc bớt môn tại bộ lọc')
      df=df.reset_index()
      st.markdown("#### 2. PHỔ ĐIỂM TRUNG BÌNH CỦA NHÓM 2")
      _w=[]
      _W=df.loc[df[index2].isin(second)]
      _h=_W
      _h=_h.set_index(index2)
      h=_h[first] #
      h=h.reset_index()
      h=h.melt(id_vars=index2,var_name='Object',value_name='Scores')
      g = sns.FacetGrid(h, col=index2,row='Object',height=10,aspect=1.5)
      g.map(sns.histplot, "Scores")
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot()
      st.markdown('**Ghi chú:**')
      st.markdown('- Biểu đồ so sánh điểm các môn đã chọn theo lớp học')
      st.markdown('- Cột: Các lớp đã chọn')
      st.markdown('- Hàng: Các môn đã chọn ở mục 1')
      st.markdown('- Có thể tùy chọn môn, thêm hoặc bớt môn tại bộ lọc')
def abc(df,index1,index2,index3,number,first_cols,second_cols,_pass):
  ds_df=number.reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
  if _pass==0:
      st.error('Vui lòng chọn số điểm qua môn để tiếp tục')
  else:
    row0_1, row0_2 = st.beta_columns(
    (1, 1))
    with row0_1:
      st.markdown('#### 3. TỈ LỆ GIỮA SINH VIÊN ĐẠT VÀ CHƯA ĐẠT CÁC MÔN NĂM 1')

      fig, ax = plt.subplots()
      first_df=df[first_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
      first_df['STATUS']=first_df['Scores'].apply(lambda x:'Đậu' if x>=_pass else 'Rớt')
      # status=first_df
      first_df=first_df[[index2,'STATUS']]
      _first_df=first_df
      _first_df=_first_df.value_counts()
      _first_df=_first_df.reset_index()
      _first_df=_first_df.pivot(index=index2,columns='STATUS',values=0)
      stacked_data2 = _first_df.apply(lambda x: x*100/sum(x), axis=1)
      st.set_option('deprecation.showPyplotGlobalUse', False)

      stacked_data2.plot(kind="bar", stacked=True,color={"Rớt": "orange", "Đậu": "blue"},fontsize = 50,figsize=(20, 20))
      plt.xlabel("Lớp")
      plt.ylabel("%")
      # ax=_second_df.plot.bar(stacked=True)
      st.pyplot()

    with row0_2:
      st.markdown("#### 4. TỈ LỆ SINH VIÊN ĐẠT VÀ CHƯA ĐẠT CÁC MÔN NĂM 2")
      fig, ax = plt.subplots()
      second_df=df[second_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
      second_df['STATUS']=second_df['Scores'].apply(lambda x:'Đậu' if x>=_pass else 'Rớt')
      # status=first_df
      second_df=second_df[[index2,'STATUS']]
      _second_df=second_df
      _second_df=_second_df.value_counts()
      _second_df=_second_df.reset_index()
      _second_df=_second_df.pivot(index=index2,columns='STATUS',values=0)
      stacked_data = _second_df.apply(lambda x: x*100/sum(x), axis=1)
      st.set_option('deprecation.showPyplotGlobalUse', False)

      stacked_data.plot(kind="bar", stacked=True,color={"Rớt": "orange", "Đậu": "blue"},fontsize = 50,figsize=(20, 20))
  
      plt.xlabel("Lớp")
      plt.ylabel("%")
      # ax=_second_df.plot.bar(stacked=True)
      st.pyplot()
      ds_df[index3] = ds_df[index3].apply(lambda x: 'Đang học' if  pd.isnull(x) else 'Nghỉ học')
      status=ds_df
      terminate=status[[index2,index3]]
      _ter=terminate.value_counts()
      _ter=_ter.reset_index()
      # a=a.reset_index()
    st.markdown('**Ghi chú:**')
    st.markdown('- So sánh tỉ lệ Đậu/Rớt các môn học năm nhất và năm 2 của các lớp')
    st.markdown('- Có thể tùy chọn môn, thêm hoặc bớt môn tại bộ lọc')
    st.markdown('#### 5. TỈ LỆ SINH VIÊN ĐANG THEO HỌC VÀ ĐÃ NGHỈ HỌC')
    ter=_ter.pivot(index=index2,columns=index3,values=0)
    stacked_data3 = ter.apply(lambda x: x*100/sum(x), axis=1)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    stacked_data3.plot(kind="bar", stacked=True,color={"Nghỉ học": "orange", "Đang học": "blue"},figsize=(25, 10))
    plt.xlabel("Lớp")
    plt.ylabel("%")
    st.pyplot()
    # _ter.plot(x=index2, y=['Đã thôi học','Đang theo học'], kind="bar")
    # st.pyplot()
    #warning list 1
    st.markdown('ĐỂ XUẤT DANH SÁCH SINH VIÊN ĐẬU/RỚT. HÃY CHỌN ĐIỀU KIỆN TẠI MỤC B TRÊN SIDEBAR')

def out(df,index1,index2,index3,numerical_cols,first_cols,second_cols,_pass):
      st.sidebar.markdown('B. TÌM SINH VIÊN RỚT NĂM 1')
      need=st.sidebar.multiselect('Đậu các môn bắt buộc',                      
                          numerical_cols.columns.tolist())
      warning_list1=[]

      if not need:
        st.sidebar.error('Vui lòng bổ sung trường thông tin tại sidebar')
      else:
        total=st.sidebar.number_input('Hoặc đạt tổng số môn cần đạt:',step=1)
        if not total:
          st.sidebar.error('Vui lòng bổ sung trường thông tin tại sidebar')
        else:
          first_df=df[first_cols].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
        first_fail=first_df.loc[(first_df.Scores<_pass)].reset_index()

        fail_O1=df
        fail_O2=df
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

        all=first_cols + second_cols
        co=len(all)
        pass_df=df[all].reset_index().melt(id_vars=[index1,index2,index3],var_name='Object',value_name='Scores')
        _pass_df=pass_df.loc[(pass_df.Scores>=_pass)]
        _pass_count=_pass_df.groupby(_pass_df[index1]).Scores.count()
        list2=_pass_count.loc[_pass_count.values >=co]
        list2=list2.reset_index()
        list22=pd.DataFrame(list2[index1])
        l22=list22.merge(df,how='left',on=index1)
        se=len(second_cols)
        pass_scond=_pass_count.loc[_pass_count.values >=se]
        list3=pass_scond.reset_index()
        list33=pd.DataFrame(list3[index1])
        l33=list33.merge(df,how='left',on=index1)
        #list study BSEM
        st.markdown('### 6. DANH SÁCH SINH VIÊN THUỘC CÁC NHÓM')
        cho=st.selectbox('Vui lòng chọn',['SINH VIÊN NĂM NHẤT CẦN ĐƯỢC CẢNH BÁO','SINH VIÊN ĐƯỢC HỌC MÔN CHUYÊN NGÀNH','SINH VIÊN HOÀN THÀNH CHƯƠNG TRÌNH NĂM 1 VÀ NĂM 2'])
        file=warning_list1
        if cho=='SINH VIÊN NĂM NHẤT CẦN ĐƯỢC CẢNH BÁO':
          file=warning_list1
        elif cho=='SINH VIÊN ĐƯỢC HỌC MÔN CHUYÊN NGÀNH':
          file=l22
        else:
          file=l33
        file
        tmp_download_link = download_link(file, 'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
      
def main():


  st.markdown("<p style='text-align: center;'><strong><span style='font-size: 28px; font-family: Arial, Helvetica, sans-serif;color:orange'>ỨNG DỤNG</span></strong></p><p style='text-align: center;'><span style='font-family: Arial, Helvetica, sans-serif;'><span style='font-size: 28px;color: orange'><strong>HỖ TRỢ QUẢN L&Yacute; TRONG GI&Aacute;O DỤC</strong></span></span></p>", unsafe_allow_html=True)
  st.markdown("<p style='text-align: right;'><strong><em><span style='font-size: 12px;'>Tạo bởi: Atom Fair - Nhóm 8 (</span></em></strong><span style='font-family: Arial, Helvetica, sans-serif;'><span style='font-size: 12px;'><em><strong>Lâm Hiếu -&nbsp;</strong></em></span></span><span style='font-family: Arial, Helvetica, sans-serif;'><span style='font-size: 12px;'><em><strong>Toàn Trần </strong></em></span></span><strong><em><span style='font-size: 12px; font-family: Arial, Helvetica, sans-serif;'> - Hạnh Nguyễn)</span></em></strong></p>", unsafe_allow_html=True)
  

  files = st.file_uploader("Tải file chứa điểm của lớp", type=['csv','xlsx','pickle'],accept_multiple_files=False)
  try:
    if not files:
          st.markdown('#### I. GIỚI THIỆU')
          st.markdown("")
          st.markdown('Operation4edu là công cụ phân tích điểm học tập của sinh viên. Cung cấp cho người dùng những thông tin quan trọng về tình hình học tập của sinh viên. Từ đó giúp giảng viên và bộ phận quản lý đưa ra những quyết định trong vận hành.')
          st.markdown('')
          st.markdown('Công cụ cho phép người dùng tải file định dạng csv hoặc excel chứa điểm của học viên. Với các trường tùy chọn thông tin tại bộ lọc sidebar để tự động truy xuất báo cáo.')
          st.markdown('Mục tiêu dự án hướng tới đối tượng sử dụng là các giảng viên/giáo viên/bộ phận quản lý đào tạo')
          st.markdown('NỘI DUNG:')
          st.markdown('- Operation Dashboard: Những chỉ số về điểm theo tổng quan')
          st.markdown('- Student checking: Xem điểm chi tiết các môn của học viên')
          st.markdown('- Phân nhóm học tập: Chia nhóm theo mô hình clustering')
          st.markdown('Dự án nằm trong chương trình Atom-Fair của DataCracy.')
          st.markdown('#### B. HƯỚNG DẪN ĐỊNH DẠNG FILE')
          st.markdown("")
          st.markdown('#### 1. Các trường thông tin bắt buộc')
          st.markdown('')
          st.markdown('- Mã số học viên: Mỗi học viên chỉ có 1 mã số học viên duy nhất')
          st.markdown(' - Tên học viên')
          st.markdown('- Lớp học: Tên lớp của học viên')
          st.markdown('- Tình trạng nghỉ học: Chứa thông tin các học viên đã nghỉ học')
          st.markdown(' - Điểm số các môn: Mỗi môn tương ứng với 1 cột')
 
          st.markdown('Dưới đây là hình ảnh minh họa 1 file đúng định dạng:')
          img1 = mpimg.imread('Capture.PNG')
          st.image(img1)

          st.markdown('#### C. MỜI BẠN BẮT ĐẦU SỬ DỤNG ỨNG DỤNG')
          st.write("Hãy tải lên 1 file định dạng .csv or .xlsx để bắt đầu xem báo cáo")

    else:
      df = get_df(files)
      choose=st.sidebar.selectbox('Chọn nội dung báo cáo:',['Operation Dashboard','Student checking','Phân nhóm học tập'])
      if choose=='Operation Dashboard':
        st.markdown('Báo cáo hiển thị kết quả học tập lớp theo môn học. Tùy chỉnh lựa chọn các môn tại bộ lọc để xem kết quả.')
        st.sidebar.markdown('A. XÁC ĐỊNH TRƯỜNG THÔNG TIN')
        index=st.sidebar.multiselect('Chọn cột chứa Mã sinh viên - Tên lớp - Tình trạng:',
                        df.columns.tolist())
        index1=0
        index2=0
        index3=0
        index1=index[0]                 
        index2=index[1]
        index3=index[2]
        df=df.set_index([index1,index2,index3]) 
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerical_cols = df.select_dtypes(include=numerics)
        number=numerical_cols
        st.sidebar.markdown('B. CHỌN NỘI DUNG ĐỂ XEM BÁO CÁO')

        sl=st.sidebar.selectbox('',['PHỔ ĐIỂM','XUẤT BÁO CÁO'])
        if sl=='PHỔ ĐIỂM':
          _df=df.reset_index()
          first=st.sidebar.multiselect('Chọn tất cả các môn để xem phổ điểm',
                      numerical_cols.columns.tolist())
          second=st.sidebar.multiselect('Chọn tất cả các lớp để so sánh',
                      _df[index2].unique().tolist())

          transform(df,first,index2,second)
        else:
          first_cols=st.sidebar.multiselect('Chọn tất cả các môn năm nhất để xem phổ điểm',
                        numerical_cols.columns.tolist())
          second_cols=st.sidebar.multiselect('Chọn tất cả các môn năm hai để xem phổ điểm',
                      numerical_cols.columns.tolist())
          _pass=st.sidebar.number_input('Mức điểm qua môn:', step=1)

          abc(df,index1,index2,index3,number,first_cols,second_cols,_pass)
          out(df,index1,index2,index3,numerical_cols,first_cols,second_cols,_pass)
      elif choose=='Student checking':
          st_id=st.sidebar.text_input('Nhập mã sinh viên:',"")
          if not st_id:
            st.sidebar.error('Vui lòng nhập mã số sinh viên')
          else:
            check_student(df,st_id)
            
      elif choose == 'Phân nhóm học tập':
          clustering(df)
  except:
    st.error('Vui lòng nhập các thông tin tại sidebar để tiếp tục')
  

main()


