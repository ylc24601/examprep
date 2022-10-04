import streamlit as st
import pandas as pd
import base64
import shutil
from io import BytesIO, StringIO
from PyPDF2 import PdfFileReader, PdfFileWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle


def fillTestSheet(uploaded_file, preview, page_num_to_trim, ID_LEFT, ID_HEIGHT, NAME_LEFT, NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT):
    packet = BytesIO()
    # create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=A4)
    can.setFont('Microsoft Jhenghei', 12)
    can.setFillColor("darkblue")
    can.drawString(ID_LEFT * mm, ID_HEIGHT * mm, str(ID))
    can.drawString(NAME_LEFT * mm, NAME_HEIGHT * mm, Name)
    can.drawString(CLASS_LEFT * mm, CLASS_HEIGHT * mm, Class)
    can.drawString(SEAT_LEFT * mm, SEAT_HEIGHT * mm, f'Seat：{Seat}')
    can.save()
    # move to the beginning of the StringIO buffer
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    # read your existing PDF
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
    # embed streamlit docs in a streamlit app
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
    # tested the line below but fail
    # pdf_display = components.iframe(f"data:application/pdf;base64,{base64_pdf}", scrolling=True)


st.sidebar.subheader("1. 上傳 masterTable.xlsx")
uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key=2)
st.sidebar.subheader("2. 上傳試卷pdf")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF file(s)", accept_multiple_files=True, key=3)
if uploaded_files or uploaded_mt is None:
    '''
    ### 目的：將學號與座位列印於對應版本之題目卷
    #### 說明：
    試卷檔名格式請依照 ...Ver_1.pdf

    其中數字為版本編號
    '''
if uploaded_mt is not None:
    df = pd.read_excel(uploaded_mt, index_col=0)
    version_num = df["Version"].nunique()
    st.dataframe(df)
    df_array = df.sort_values(by='Seat_index').to_numpy()
    num_rows, num_cols = df_array.shape
    col1, col2, col3 = st.columns(3)
    col1.metric(label="應考人數", value=f"{num_rows}人")
    col2.metric(label="試題版本", value=version_num)
if uploaded_files and uploaded_mt is not None:
    col3.metric(label="上傳檔案數", value=len(uploaded_files),
                delta=len(uploaded_files)-version_num)
    if len(uploaded_files) == version_num:
        with st.expander("定位參數修改"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ID_LEFT = st.number_input("頁面左邊界至學號左緣之距離(mm): ", value=30)
                ID_HEIGHT = st.number_input(
                    "頁面下邊界至學號下緣之距離(mm): ", value=281)
                # page_num_to_trim = st.number_input("刪減PDF檔倒數頁數: ", min_value=0, value=2, step=1)
                # page_num_to_trim = int(page_num_to_trim)
            with col2:
                NAME_LEFT = st.number_input(
                    "頁面左邊界至姓名左緣之距離(mm): ", value=65)
                NAME_HEIGHT = st.number_input(
                    "頁面下邊界至姓名下緣之距離(mm): ", value=281)
            with col3:
                CLASS_LEFT = st.number_input(
                    "頁面左邊界至期班左緣之距離(mm): ", value=105)
                CLASS_HEIGHT = st.number_input(
                    "頁面下邊界至期班下緣之距離(mm): ", value=281)
            with col4:
                SEAT_LEFT = st.number_input(
                    "頁面左邊界至座位左緣之距離(mm): ", value=167)
                SEAT_HEIGHT = st.number_input(
                    "頁面下邊界至座位下緣之距離(mm): ", value=287)
        st.info("目前Preview功能僅限使用Firefox")
        col1, col2 = st.columns(2)

        cognero_sheet = col2.checkbox("由Macmillan(Cognero)出題", True)
        col2.caption("勾選後會刪除最後兩頁(答案)")
        if cognero_sheet:
            page_num_to_trim = 2
        else:
            page_num_to_trim = 0
        if col1.button("Preview"):
            ID, Name, Class, Seat_index, Seat, Version = df_array[0, :]
            fillTestSheet(uploaded_files[0], True, page_num_to_trim, ID_LEFT, ID_HEIGHT,
                            NAME_LEFT, NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT)
            displayPDF("preview.pdf")
        if col1.button("Make All Sheets"):
            progress_bar = st.progress(0)
            current_progress = 0.0
            for ID, Name, Class, Seat_index, Seat, Version in df_array:
                for file in uploaded_files:
                    if 'Ver_' + str(Version) in file.name:
                        fillTestSheet(file, False, page_num_to_trim, ID_LEFT, ID_HEIGHT, NAME_LEFT,
                                        NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT)
                        current_progress += (1/(num_rows+1))
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
    else:
        st.warning("上傳檔案數與試題版本數不符！")