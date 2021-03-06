import base64
import re
import shutil
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
from pdfminer.high_level import extract_text
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from batch_fill_AS import batchFillAS

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
    randomized_seat = seat.iloc[0:len(dataframe)].sample(
        frac=1).reset_index(drop=True)
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
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
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


def makeAnnouTable(data, file='announce_table.pdf'):
    doc = SimpleDocTemplate(
        file, pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
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


def into_excel(index_output=True, **kwargs):
    if kwargs is not None:
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        for sheet_name, df in kwargs.items():
            df.to_excel(writer, index=index_output, sheet_name=sheet_name)
            # workbook = writer.book
            # worksheet = writer.sheets['Sheet1']
            # format1 = workbook.add_format({'num_format': '0.00'})
            # worksheet.set_column('A:A', None, format1)
        writer.save()
        processed_data = output.getvalue()
        return processed_data


def fillTestSheet(uploaded_file, preview, page_num_to_trim, ID_LEFT, ID_HEIGHT, NAME_LEFT, NAME_HEIGHT, CLASS_LEFT, CLASS_HEIGHT, SEAT_LEFT, SEAT_HEIGHT):
    packet = BytesIO()
    # create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=A4)
    can.setFont('Microsoft Jhenghei', 12)
    can.setFillColor("darkblue")
    can.drawString(ID_LEFT * mm, ID_HEIGHT * mm, str(ID))
    can.drawString(NAME_LEFT * mm, NAME_HEIGHT * mm, Name)
    can.drawString(CLASS_LEFT * mm, CLASS_HEIGHT * mm, Class)
    can.drawString(SEAT_LEFT * mm, SEAT_HEIGHT * mm, f'Seat???{Seat}')
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


def get_pdf_page_count(file):
    reader = PdfFileReader(BytesIO(file.getvalue()), "rb")
    return reader.getNumPages()


@st.cache
def getAnswer(uploaded_files):
    pre_list = []
    for file in files:
        page_num = get_pdf_page_count(file)
        text = extract_text(file,page_numbers=[page_num-2], codec='utf-8').replace('\xa0', ' ')
        answer_key = text.split('Answer Key')[1]
        matches = re.findall(r'\d+\.\s[A-E|a-e]', answer_key)
        column = [answer[-1] for answer in matches]
        pre_list.append(column)
    answer_list = list(zip(*pre_list))
    return answer_list


def get_original_question(file, question_num):
    page_num = get_pdf_page_count(file)
    ver = re.findall("Ver_(\d)", file.name)
    pdf = pdfplumber.open(BytesIO(file.getvalue()))
    page = pdf.pages[page_num-1]
    string = page.extract_text()
    matches = re.findall("\\n(\d{1,2})\s(\d{1,2})", string)
    df = pd.DataFrame(matches)
    df.rename(columns={df.columns[0]: "question", df.columns[1]: str(ver[0])}, inplace=True)
    df.set_index('question', inplace=True)
    return df


def answer_dataframe(file, question_num):
    data = StringIO(student_answer_file.read().decode("utf-8"))
    as_list = [[line[7:16], line[16:question_num+16]] for line in data.readlines()]
    # split student answers to a single char in a list
    conv_list = []
    for line in as_list:
        temp = []
        for char in line[1]:
            temp.append(char)
        comb = [line[0]] + temp
        conv_list.append(comb)
    df = pd.DataFrame(conv_list)
    df.dropna(inplace=True)
    df.rename(columns={0: 'ID'}, inplace=True)
    df.set_index("ID", inplace=True)
    problem_df = df[~df.isin(['A', 'B', 'C', 'D', 'E']).all(axis=1)]
    df.reset_index(inplace=True)
    return df, problem_df


def grade_cal(st_ans, mt, correct_answer, from_cognero=True, qnum=50, point=2, scramble_map=None):
    ans_array = st_ans.to_numpy()
    results = []
    detail = []
    if from_cognero:
        correctness = [0] * qnum
    # loop through every student
    progress_bar = st.progress(0)
    current_progress = 0.0
    for row in ans_array:
        score = 0
        # match the test version
        personal_detail = [row[0]]
        ver = mt[mt.ID == int(row[0])]['Version'].values
        for i in range(1, qnum+1):
            if row[i].upper() in str(correct_answer.loc[ver][i]):
                score += point
                if from_cognero:
                    original_question = scramble_map.loc[i][str(ver[0])]
                    correctness[original_question-1] += 1
                personal_detail.append('O')
            else:
                personal_detail.append('X')

        results.append([row[0], score])
        detail.append(personal_detail)
        current_progress += (1/(len(ans_array)+1))
        progress_bar.progress(current_progress)
    progress_bar.progress(1.0)
    if from_cognero:
        return results, detail, correctness
    return results, detail


st.title("Examination Workflow")
menu = ["????????????", "?????????", "?????????", "??????????????????", "????????????"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("---")


if choice == "????????????":

    st.markdown("### ???????????????Master Table??????????????????")
    st.sidebar.subheader("1. ?????????????????????")
    title = st.sidebar.text_input(
        '?????????????????????????????????', "Biochemistry 1st Exam Seating Table")

    st.sidebar.subheader("2. ?????????????????????????????????")
    seat_choice = st.sidebar.radio(
        "????????????", ('?????????306???(??????????????????)', '?????????250???(??????????????????)'))
    if seat_choice == "?????????306???(??????????????????)":
        SEAT = pd.read_excel("seat_306.xlsx", header=None, names=['Seat'])
    else:
        SEAT = pd.read_excel("seat_250.xlsx", header=None, names=['Seat'])

    version = st.sidebar.slider("????????????", 1, 10, 5)
    st.sidebar.subheader("3. ??????????????????")
    uploaded_file = st.sidebar.file_uploader("????????????: xlsx", key=1)
    if uploaded_file is None:
        roll_template = pd.read_excel("roll_list.xlsx")
        st.subheader("????????????????????????")
        # CSS to inject contained in a string
        hide_dataframe_row_index = """
                    <style>
                    .row_heading.level0 {display:none}
                    .blank {display:none}
                    </style>
                    """
        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        st.dataframe(roll_template, 300)
        roll_template_xls = into_excel(
            index_output=False, Sheet1=roll_template)
        st.download_button(
            label='???????????????',
            data=roll_template_xls,
            file_name="roll_list.xlsx")
        st.write("---")
        image = Image.open("seat_map.png")
        st.image(image, caption="Master Table?????????????????????????????????")
    else:
        roll_list = pd.read_excel(uploaded_file)
        final_output = master_table(roll_list, SEAT, version)
        st.dataframe(final_output)
        mt = into_excel(masterTable=final_output)
        makeAnnouTable(columnize(df2list(final_output), 41, 2, heading=1))

        st.write("???Excel???????????????????????????????????????????????????????????????!")
        st.download_button(
            label='Download Master Table',
            data=mt,
            file_name="masterTable.xlsx")
        st.write("???????????????????????????????????????????????????????????????")
        with open('announce_table.pdf', 'rb') as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button('Download PDF', data=PDFbyte,
                           file_name="announce_table.pdf", mime="application/octet-stream")

if choice == "?????????":
    st.sidebar.subheader("1. ??????????????????")
    as_title = st.sidebar.text_input('????????????????????????', "Biochem-1")
    st.sidebar.subheader("2. ?????? masterTable.xlsx")
    uploaded_mt = st.sidebar.file_uploader("????????????: xlsx", key=2)
    st.sidebar.subheader("3.????????????????????????????????? mm")
    ID_left = st.sidebar.number_input("???????????????????????????????????????(0)?????????: ", 20.45)
    ID_right = st.sidebar.number_input("???????????????????????????????????????(9)?????????: ", 60.47)
    ID_top = st.sidebar.number_input("??????????????????????????????????????????: ", 192.58)
    ID_bottom = st.sidebar.number_input("??????????????????????????????????????????:: ", 151.05)
    if uploaded_mt is None:
        '''
        ### ?????????????????? Master Table ???????????????????????????????????????
        ????????????????????????????????????????????????????????? **A5**

        ????????????????????????????????????????????????????????????
        '''
    else:
        df_seat = pd.read_excel(uploaded_mt, index_col=0)
        st.dataframe(df_seat)
        batchFillAS(df_seat, "AnswerSheet.pdf", as_title,
                    ID_left, ID_right, ID_top, ID_bottom)
        with open("AnswerSheet.pdf", 'rb') as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button(
            'Download PDF',
            data=PDFbyte,
            file_name="AnswerSheet.pdf",
            mime="application/octet-stream")
if choice == "?????????":
    st.sidebar.subheader("1. ?????? masterTable.xlsx")
    uploaded_mt = st.sidebar.file_uploader("????????????: xlsx", key=2)
    st.sidebar.subheader("2. ????????????pdf")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF file(s)", accept_multiple_files=True, key=3)
    if uploaded_files or uploaded_mt is None:
        '''
        ### ????????????????????????????????????????????????????????????
        #### ?????????
        ??????????????????????????? ...Ver_1.pdf

        ???????????????????????????
        '''
    if uploaded_mt is not None:
        df = pd.read_excel(uploaded_mt, index_col=0)
        version_num = df["Version"].nunique()
        st.dataframe(df)
        df_array = df.sort_values(by='Seat_index').to_numpy()
        num_rows, num_cols = df_array.shape
        col1, col2, col3 = st.columns(3)
        col1.metric(label="????????????", value=f"{num_rows}???")
        col2.metric(label="????????????", value=version_num)
    if uploaded_files and uploaded_mt is not None:
        col3.metric(label="???????????????", value=len(uploaded_files),
                    delta=len(uploaded_files)-version_num)
        if len(uploaded_files) == version_num:
            with st.expander("??????????????????"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    ID_LEFT = st.number_input("???????????????????????????????????????(mm): ", value=30)
                    ID_HEIGHT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=281)
                    # page_num_to_trim = st.number_input("??????PDF???????????????: ", min_value=0, value=2, step=1)
                    # page_num_to_trim = int(page_num_to_trim)
                with col2:
                    NAME_LEFT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=65)
                    NAME_HEIGHT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=281)
                with col3:
                    CLASS_LEFT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=105)
                    CLASS_HEIGHT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=281)
                with col4:
                    SEAT_LEFT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=167)
                    SEAT_HEIGHT = st.number_input(
                        "???????????????????????????????????????(mm): ", value=287)
            st.info("??????Preview??????????????????Firefox")
            col1, col2 = st.columns(2)

            cognero_sheet = col2.checkbox("???Macmillan(Cognero)??????", True)
            col2.caption("??????????????????????????????(??????)")
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
            st.warning("??????????????????????????????????????????")
if choice == "??????????????????":
    st.sidebar.subheader("????????????pdf???")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF file(s)", accept_multiple_files=True, key=3)
    files = sorted(uploaded_files, key=lambda x: x.name)
    st.subheader("?????????????????????Macmillan(Cognero) Test Generator???????????????")
    if len(uploaded_files) != 0:
        if st.button("Extract Answers"):
            answer_df = pd.DataFrame(getAnswer(files))
            answer_df.index += 1
            answer_df.columns += 1
            question_num = len(answer_df)
            #scramble map generation
            if len(uploaded_files) == 1:
                map_df =get_original_question(uploaded_files[0], question_num)
            else:
                df_list = []
                unique_df = []
                # map_df = pd.DataFrame({"question": range(1, 51)})
                for file in files:
                    df = get_original_question(file, question_num)
                    df_list.append(df)
                    unique_df.append(df.nunique())
                map_df=pd.concat(df_list, axis=1)
                # map_df.set_index("question", inplace=True)
            col1, col2 = st.columns(2)
            col1.subheader("Correct Answers")
            col1.dataframe(answer_df)
            col2.subheader("Scramble Map")
            col2.dataframe(map_df)
            st.write("???????????????????????????????????????????????????")
            # st.write(unique_df)
            answer_xls = into_excel(answers=answer_df, map=map_df)
            excel_clicked = st.download_button(
                label='Download Excel File',
                data=answer_xls,
                file_name="correct_answers.xlsx")
if choice == "????????????":
    # Parameter Setting
    st.subheader("????????????")
    col1, col2, col3 = st.columns(3)

    qnum = col1.number_input(label="??????", value=50)
    point = col2.number_input(label="????????????", value=2.0)
    col3.metric(label="??????", value = qnum * point)
    from_cognero = col1.checkbox(label="???Macmillan????????????", value=True)
    if qnum * point != 100:
        st.warning("??????: ????????????100???!")
    st.write("---")
    # Sidebar Setting
    st.sidebar.subheader("?????? masterTable.xlsx")
    uploaded_mt = st.sidebar.file_uploader("????????????: xlsx", key=2)
    st.sidebar.subheader("??????????????????????????????????????????")
    student_answer_file = st.sidebar.file_uploader(
        "Upload txt file", accept_multiple_files=False, key=4)
    st.sidebar.subheader("??????????????????")
    correct_answers = st.sidebar.file_uploader("????????????: xlsx", key=5)

    if uploaded_mt is not None:
        df = pd.read_excel(uploaded_mt, index_col=0)
        version_num = df["Version"].nunique()
        st.subheader("Master Table")
        st.dataframe(df)
    if student_answer_file is not None:
        st_ans_df, problem_df = answer_dataframe(student_answer_file, qnum)
        st.subheader("????????????")
        st.dataframe(st_ans_df)
        st.subheader("????????????")
        st.dataframe(problem_df)
    if correct_answers is not None:
        ca_df = pd.read_excel(correct_answers, index_col=0).T
        ca_df = ca_df.apply(lambda x: x.str.upper())
        st.subheader("????????????")
        st.dataframe(ca_df)
        xl = pd.ExcelFile(correct_answers)
        sheets_num = len(xl.sheet_names)
        if sheets_num == 2:
            scramble_map = pd.read_excel(
                correct_answers, index_col=0, sheet_name=1, header=0)
            st.subheader("?????????????????????")
            st.dataframe(scramble_map)
    col1, col2, col3, col4, col5 = st.columns(5)
    if uploaded_mt is not None:
        col1.metric(label="Master Table??????", value=len(df))
        col4.metric(label="Master Table?????????", value=df.Version.nunique())
    if student_answer_file is not None:
        col2.metric(label="???????????????", value=len(st_ans_df))
        col3.metric(label="??????????????????", value=len(problem_df))
    if correct_answers is not None:
        col5.metric(label="?????????????????????", value=len(ca_df))
    if student_answer_file and correct_answers and uploaded_mt is not None:
        if df.Version.nunique() != len(ca_df):
            st.warning("??????! ???????????????")
        if len(df) != len(st_ans_df):
            st.warning("??????! ?????????????????????????????????")

        if st.button(label="Calculate"):
            if not from_cognero:
                scramble_map = None
            results = grade_cal(st_ans_df, df, ca_df, from_cognero,
                                qnum=qnum, point=point, scramble_map=scramble_map)
            result_df = pd.DataFrame(results[0], columns=('ID', 'Score'))
            detail_df = pd.DataFrame(results[1])
            col1, col2 = st.columns(2)
            col1.subheader("????????????")
            col1.dataframe(result_df)
            if len(results) == 3:
                correctness_df = pd.DataFrame(
                    results[2], index=range(1, qnum+1), columns=("correct_num",))
                correctness_df["Percent"] = correctness_df['correct_num'] * \
                    100/len(result_df)
                correctness_df = correctness_df.round(1)
                col2.subheader("???????????????")
                col2.dataframe(correctness_df)
            st.subheader("????????????????????????")
            st.dataframe(detail_df)
            st.subheader("????????????")
            # Create distplot with custom bin_size
            # group_labels = ['Grade Distribution']
            # fig = ff.create_distplot([result_df.Score], group_labels, bin_size=1)

            # # Plot!
            # st.plotly_chart(fig, use_container_width=True)
            st.write(result_df.describe().T)

            if from_cognero:
                score_xls = into_excel(score=result_df, detail=detail_df,
                                       correctness=correctness_df, stats=result_df.describe())
            else:
                score_xls = into_excel(
                    score=result_df, detail=detail_df, stats=result_df.describe())
            csv = convert_df(result_df)
            col1, col2, col3 = st.columns(3)
            col1.download_button(
                label='Download Excel File',
                data=score_xls,
                file_name="Score.xlsx")
            col2.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name='score.csv',
                mime='text/csv',)
            col2.write("??????????????????????????????")
