import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm, inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import streamlit as st
import os
import base64
# ------------------------------------------

pdfmetrics.registerFont(TTFont('Microsoft Jhenghei', 'Microsoft Jhenghei.ttf'))
width, height = A4
# uncomment the line below when run in local
# os.chdir("/Users/YLC/SynologyDrive/Code/Code_for_Exam/st_version")

@st.cache
def master_table(dataframe, seat, ver_num):
    dataframe['ID'] = dataframe['ID'].apply(str)
    seat.index.name = 'Seat_index'
    seat.reset_index(inplace=True)
    # Generate test sheet version
    seat['Version'] = np.arange(len(seat)) % ver_num + 1
    randomized_seat = seat.iloc[0:len(dataframe)].sample(frac=1).reset_index(drop=True)
    # Combine df and randomized_seat
    df_seat = pd.concat([dataframe, randomized_seat], join='inner', axis=1)
    return df_seat
    # Dataframe for Seat Announcement

@st.cache
def df2list(df):
    seat_annou = df[['ID', 'Name', 'Seat']]
    seat_annou.index += 1
    # reset column of the Dataframe
    seat_annou = seat_annou.T.reset_index().T.reset_index(drop=False)
    # for pdf generation
    return seat_annou.values.tolist()


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def myFirstPage(canvas, doc):
    canvas.saveState()
    canvas.setFont('Microsoft Jhenghei', 16)
    canvas.drawCentredString(width / 2, height - 0.4 * inch, title)
    canvas.setFont('Microsoft Jhenghei', 9)
    canvas.drawCentredString(width / 2, 0.25 * inch, "Page %d" % doc.page)
    canvas.restoreState()


def myLaterPages(canvas, doc):
    canvas.saveState()
    canvas.setFont('Microsoft Jhenghei', 16)
    canvas.drawCentredString(width / 2, height - 0.4 * inch, title)
    canvas.setFont('Microsoft Jhenghei', 9)
    canvas.drawCentredString(width / 2, 0.25 * inch, "Page %d" % doc.page)
    canvas.restoreState()


def makeAnnouTable(data, file = 'announce_table.pdf'):
    doc = SimpleDocTemplate(file, pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    # container for the "Flowable" objects
    elements = []
    # convert data into table with each column width
    tableThatSplitsOverPages = Table(data, [1.2 * cm, 2.5 * cm, 2 * cm, 2 * cm,
                                            1.2 * cm, 2.5 * cm, 2 * cm, 2 * cm], repeatRows=1)
    tableThatSplitsOverPages.hAlign = 'LEFT'
    tblStyle = TableStyle([('FONT', (0, 0), (-1, -1), 'Microsoft Jhenghei'),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('LINEBELOW', (0, 0), (-1, -1), 1, colors.black),
                           ('BOX', (0, 0), (3, -1), 1, colors.grey),
                           ('BOX', (0, 0), (-1, -1), 1.5, colors.black)])
    tblStyle.add('BACKGROUND', (0, 0), (-1, 0), colors.darkslategrey)
    tblStyle.add('BACKGROUND', (0, 1), (-1, -1), colors.white)
    tableThatSplitsOverPages.setStyle(tblStyle)
    elements.append(tableThatSplitsOverPages)
    doc.build(elements, onFirstPage=myFirstPage, onLaterPages=myLaterPages)


def columnize(data, rows, cols, heading=0):
    if heading == 1:
        transformed_list = [data[0] * cols]
    else:
        transformed_list = []
    for step in range(heading, len(data) - 1, rows * cols):
        for row in range(rows):
            temp = []
            for col in range(cols):
                if step + row + col * rows < len(data):
                    temp += data[step + row + col * rows]
            if len(temp) != 0:
                transformed_list.append(temp)
    return transformed_list


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

st.write("## 製作試場座位表")
st.sidebar.subheader("1. 考試名稱")
title = st.sidebar.text_input('顯示於座位公告表之標題', "Biochemistry 1st Exam Seat Table")
st.sidebar.write("---")
st.sidebar.subheader("2. 上傳學生名冊")
uploaded_file = st.sidebar.file_uploader("檔案格式: xlsx")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    roll_list = pd.read_excel(uploaded_file)
    # st.dataframe(roll_list, 300)
st.sidebar.write("---")
st.sidebar.subheader("3. 選擇試場座位與試卷版本")
choice = st.sidebar.radio("座位安排", ('致德堂306人(坐二排空一排)', '致德堂250人(坐一排空一排)'))
if choice == "致德堂306人(坐二排空一排)":
    SEAT = pd.read_excel("seat_306.xlsx", header=None, names=['Seat'])
else:
    SEAT = pd.read_excel("seat_250.xlsx", header=None, names=['Seat'])

version = st.sidebar.slider("考卷版本", 1, 10, 1)

if uploaded_file is None:
    "**還沒上傳學生名冊!**"
else:
    final_output = master_table(roll_list, SEAT, version)
    st.dataframe(final_output)
    csv = convert_df(final_output)
    makeAnnouTable(columnize(df2list(final_output), 41, 2, heading=1))
    with open('announce_table.pdf', 'rb') as pdf_file:
        PDFbyte = pdf_file.read()
    st.write("csv為重要檔案，請妥善保存!")
    csv_clicked = st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="masterTable.csv",
        mime='text/csv'
    )
    st.write("PDF印出至少三份張貼於試場外")
    st.download_button('Download PDF', data=PDFbyte, file_name="st_version/announce_table.pdf", mime="application/octet-stream")