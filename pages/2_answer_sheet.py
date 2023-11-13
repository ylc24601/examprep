import streamlit as st
import pandas as pd
from batch_fill_AS import batchFillAS

st.sidebar.subheader("1. 輸入考試名稱")
as_title = st.sidebar.text_input('印於答案卡之標題')
st.sidebar.subheader("2. 上傳 masterTable.xlsx")
uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key=2)
st.sidebar.subheader("3.設定學號列印位置，單位 mm")
ID_left = st.sidebar.number_input("答案卡左側邊緣至學號左邊界(0)之距離: ", value=20.45)
ID_right = st.sidebar.number_input("答案卡左側邊緣至學號右邊界(9)之距離: ", value=60.47)
ID_top = st.sidebar.number_input("答案卡下緣至學號上邊界之距離: ", value=192.58)
ID_bottom = st.sidebar.number_input("答案卡下緣至學號下邊界之距離:: ", value=151.05)
if uploaded_mt is None:
    '''
    ### 自動依 Master Table 將學號與座位列印於答案卡上
    1. 建議在固定的印表機列印，紙張設定須改為 **A5**
    2. 確認列印選項中的縮放比例無調整(100%)
    3. 先列印一張，調整參數確認無誤後再印出全部
    '''

else:
    if as_title == "":
        st.warning("記得輸入考試標題")
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