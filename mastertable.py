import numpy as np
import pandas as pd
import shutil
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import mm, cm, inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from io import BytesIO
import streamlit as st
import os
import base64
from PIL import Image
from batch_fill_AS import batchFillAS
from PyPDF2 import PdfFileWriter, PdfFileReader
# ------------------------------------------

pdfmetrics.registerFont(TTFont('Microsoft Jhenghei', 'Microsoft Jhenghei.ttf'))
width, height = A4


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
    return df.to_excel(engine='xlsxwriter')

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


# def create_download_link(val, filename):
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def fillTestSheet(uploaded_file, preview, page_num_to_trim, ID_LEFT, ID_HEIGHT, NAME_LEFT, NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT):
    packet = BytesIO()
    # create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=A4)
    can.setFont('Microsoft Jhenghei', 12)
    can.drawString(ID_LEFT * mm, ID_HEIGHT * mm, str(ID))
    can.drawString(NAME_LEFT * mm, NAME_HEIGHT * mm, Name)
    can.drawString(CLASS_LEFT * mm, CLASS_HEIGHT * mm, Class)
    can.drawString(SEAT_LEFT * mm, SEAT_HEIGHT * mm, f'Seat：{Seat}')
    can.save()
    # move to the beginning of the StringIO buffer
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    # read your existing PDF
    # bytes_data = file.read()
    # print(bytes_data)
    existing_pdf = PdfFileReader(BytesIO(uploaded_file.getvalue()), "rb")
    output = PdfFileWriter()
    # add the "watermark" (which is the new pdf) on the existing page
    page = existing_pdf.getPage(0)
    page.mergePage(new_pdf.getPage(0))
    output.addPage(page)

    for pageNum in range(1, existing_pdf.numPages - page_num_to_trim):
        page_obj = existing_pdf.getPage(pageNum)
        output.addPage(page_obj)

    # finally, write "output" to a real file
    if preview:
        output_stream = open(f'preview.pdf', "wb")
    else:
        output_stream = open(f'./tmp/{Seat}.pdf', "wb")
    output.write(output_stream)
    output_stream.close()


def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


menu = ["試場座位", "答案卡", "試題卷", "匯整正確答案", "成績計算"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("---")


if choice == "試場座位":
    st.title("試務工作流程-步驟一")
    st.write("## 製作試場座位表")
    st.sidebar.subheader("1. 考試名稱")
    title = st.sidebar.text_input('顯示於座位公告表之標題', "Biochemistry 1st Exam Seat Table")
    st.sidebar.subheader("2. 上傳學生名冊")
    uploaded_file = st.sidebar.file_uploader("檔案格式: xlsx", key = 1)
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        roll_list = pd.read_excel(uploaded_file)
        # st.dataframe(roll_list, 300)

    st.sidebar.subheader("3. 選擇試場座位與試卷版本")
    seat_choice = st.sidebar.radio("座位安排", ('致德堂306人(坐二排空一排)', '致德堂250人(坐一排空一排)'))
    if seat_choice == "致德堂306人(坐二排空一排)":
        SEAT = pd.read_excel("seat_306.xlsx", header=None, names=['Seat'])
    else:
        SEAT = pd.read_excel("seat_250.xlsx", header=None, names=['Seat'])

    version = st.sidebar.slider("考卷版本", 1, 10, 1)

    if uploaded_file is None:
        st.empty()
    else:
        final_output = master_table(roll_list, SEAT, version)
        st.dataframe(final_output)
        mt = to_excel(final_output)
        makeAnnouTable(columnize(df2list(final_output), 41, 2, heading=1))

        st.write("此為重要檔案，請妥善保存!")
        csv_clicked = st.download_button(
            label='Download Excel File',
            data=mt,
            file_name="masterTable.xlsx")
        st.write("PDF印出至少三份張貼於試場外")
        with open('announce_table.pdf', 'rb') as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button('Download PDF', data=PDFbyte, file_name="announce_table.pdf", mime="application/octet-stream")
if choice == "答案卡":
    st.title("試務工作流程-步驟二")
    st.sidebar.subheader("1. 考試名稱")
    as_title = st.sidebar.text_input('顯示於答案之標題', "Biochem-1")
    st.sidebar.subheader("2. 上傳 masterTable.xlsx")
    uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key = 2)
    st.sidebar.subheader("3.設定列印位置，單位 mm")
    ID_left = st.sidebar.number_input("答案卡左側邊緣至學號左邊界(0)之距離: ", 20.45)
    ID_right = st.sidebar.number_input("答案卡左側邊緣至學號右邊界(9)之距離: ", 60.47)
    ID_top = st.sidebar.number_input("答案卡下緣至學號上邊界之距離: ", 192.58)
    ID_bottom = st.sidebar.number_input("答案卡下緣至學號下邊界之距離:: ", 151.05)
    if uploaded_mt is not None:
        df_seat = pd.read_excel(uploaded_mt, index_col=0)
        st.dataframe(df_seat)
        batchFillAS(df_seat, "AnswerSheet.pdf", as_title, ID_left, ID_right, ID_top, ID_bottom)
        with open("AnswerSheet.pdf", 'rb') as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button('Download PDF', data=PDFbyte, file_name="AnswerSheet.pdf", mime="application/octet-stream")
if choice == "試題卷":
    st.sidebar.subheader("1. 上傳 masterTable.xlsx")
    uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key = 2)
    st.sidebar.subheader("2. 上傳試卷pdf")
    uploaded_files = st.sidebar.file_uploader("Upload PDF file(s)", accept_multiple_files=True, key=3)
    if uploaded_mt is not None:
        df = pd.read_excel(uploaded_mt, index_col=0)
        st.dataframe(df)
        df_array = df.sort_values(by='Seat_index').to_numpy()
        num_rows, num_cols = df_array.shape
    if uploaded_files and uploaded_mt is not None:
        with st.expander("定位參數修改"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ID_LEFT = st.number_input("頁面左邊界至學號左緣之距離(mm): ", value=30)
                ID_HEIGHT = st.number_input("頁面下邊界至學號下緣之距離(mm): ", value=281)
                page_num_to_trim = st.number_input("刪減PDF檔倒數頁數: ", min_value=0, value=2, step=1)
            with col2:
                NAME_LEFT = st.number_input("頁面左邊界至姓名左緣之距離(mm): ", value=65)
                NAME_HEIGHT = st.number_input("頁面下邊界至姓名下緣之距離(mm): ", value=281)
            with col3:
                CLASS_LEFT = st.number_input("頁面左邊界至期班左緣之距離(mm): ", value=105)
                CLASS_HEIGHT = st.number_input("頁面下邊界至期班下緣之距離(mm): ", value=281)
            with col4:
                SEAT_LEFT = st.number_input("頁面左邊界至座位左緣之距離(mm): ", value=167)
                SEAT_HEIGHT = st.number_input("頁面下邊界至座位下緣之距離(mm): ", value=287)
        if st.button("Preview"):
            ID, Name, Class, Seat_index, Seat, Version = df_array[0,:]
            fillTestSheet(uploaded_files[0],True, page_num_to_trim, ID_LEFT, ID_HEIGHT, NAME_LEFT, NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT)
            displayPDF("preview.pdf")
        if st.button("Make All Sheets"):
            progress_bar = st.progress(0)
            current_progress = 0.0
            for ID, Name, Class, Seat_index, Seat, Version in df_array:
                for file in uploaded_files:
                    if 'Ver_' + str(Version) in file.name:
                        fillTestSheet(file, False, page_num_to_trim, ID_LEFT, ID_HEIGHT, NAME_LEFT, NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT)
                        current_progress += (1/(num_rows))
                        progress_bar.progress(current_progress)
            shutil.make_archive("archive", 'zip', "./tmp")
            progress_bar.progress(1.0)
            st.success("Done!")
            st.balloons()
            with open("archive.zip", "rb") as zf:
                btn = st.download_button(
                label='Download ZIP',
                data=zf,
                file_name="archive.zip",
                mime="application/zip")

if choice == "匯整正確答案":
    st.write("施工中")
    image = Image.open("under_construction.gif")
    st.image(image)
if choice == "成績計算":
    st.write("施工中")
    image = Image.open("under_construction.gif")
    st.image(image)
