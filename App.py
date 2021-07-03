import streamlit as st
import requests #-> Để gọi API
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
import requests
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


#td_df
sh2=gc1.open("Group 8 - Atom Fair - Dataset").worksheet('Results')
results=sh2.get_all_records()
results=pd.DataFrame(results)
results

#CREATED SEND MAIL BOT

import email, smtplib, ssl # to automate email
import email as mail
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests, json # to pull API, and work with json
import datetime as dt # to work with date, time
from bs4 import BeautifulSoup # to work with web scrapping (HTML)
from IPython.core.display import HTML # to display HTML in the notebook

subject = st.text_input('Enter your subject: ')#INPUT1: Subject of the Email
receiver_email = st.text_input("Enter receiver email: ") #INPUT2: email address to receive the email
sender_email = st.text_input('SENDER_EMAIL:')
password = st.text_input('PWD_EMAIL')



# (1) Create the email head (sender, receiver, and subject)
email = MIMEMultipart()
email["From"] = sender_email
email["To"] = receiver_email 
email["Subject"] = subject


# (2) Nội dung email
# Có thể tùy chỉnh nội dung bằng cách thay đổi text trong đoạn này
html1 = """
    <html>
    <h1><strong>Hello there</strong></h1>
    <body>
    <p>This email sent from the code I learned!<br>
       So happy and want to share with you!<br>
       
    </p>
    </body>
    </html>
    """
html2 = """
<html>
Email sent at <b>{}</b><br>
</html>
""".format(dt.datetime.now().isoformat())

text3 = '--- End ----' # lưu ý đây là dữ liệu dạng chuỗi (string)


# Combine parts
part1 = MIMEText(html1, 'html')
part2 = MIMEText(html2, 'html')
part3 = MIMEText(text3, 'plain')


def send_email(receiver_email, subject, part1, part2, part3):
        for i in range(0,len(receiver_email)+1):
            if i<len(receiver_email):
            # (1) Create the email head (sender, receiver, and subject)
                email = MIMEMultipart()
                email["From"] = sender_email
                email["To"] = receiver_email[i]
                email["Subject"] = subject

                email.attach(part1)
                email.attach(part2)
                email.attach(part3)

                # (3) Create SMTP session for sending the mail
                session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
                session.starttls() #enable security
                session.login(sender_email, password) #login with mail_id and password
                text = email.as_string()
                session.sendmail(sender_email, receiver_email, text)
                session.quit()

                print('DONE! Mail Sent'.format(sender_email, receiver_email))
                break
send_email(receiver_email, subject, part1, part2, part3)