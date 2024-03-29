from io import BytesIO
import pandas as pd
import streamlit as st
import pdfplumber
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader
import re
from master_table import into_excel


def get_pdf_page_count(file):
    reader = PdfReader(BytesIO(file.getvalue()), "rb")
    return len(reader.pages)


@st.cache_data
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


st.sidebar.subheader("上傳試卷pdf檔")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF file(s)", accept_multiple_files=True, key=3)
files = sorted(uploaded_files, key=lambda x: x.name)
st.subheader("此功能僅限於以Macmillan(Cognero) Test Generator出題之試卷")
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
        st.write("下載前請檢查試卷版本與答案是否相符")
        st.write('如有送分題，可在Excel中全選答案儲存格後以conditional formatting加入公式')
        st.write('`=ROW()=MATCH("36",map!B:B,0)`，其中36是範例原始題號')
        st.write('將要送分的答案改為ABCDE')
        # st.write(unique_df)
        answer_xls = into_excel(answers=answer_df, map=map_df)
        excel_clicked = st.download_button(
            label='Download Excel File',
            data=answer_xls,
            file_name="correct_answers.xlsx")