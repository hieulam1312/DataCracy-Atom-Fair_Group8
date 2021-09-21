from numpy import histogram
import streamlit as st
import pandas as pd
sample_ID=[st.text_area('Scan Mẫu')]
step=[st.selectbox('Chọn thao tác',['Trả mẫu','Mượn mẫu'])]

history=pd.DataFrame(list(zip(sample_ID, step)),
               columns =['Tên Mẫu', 'Thao tác'])

st.write(history)

