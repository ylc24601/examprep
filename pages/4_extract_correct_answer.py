from io import BytesIO
import re
import pandas as pd
import streamlit as st
import pdfplumber
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader
from master_table import into_excel


# ---------------------------
# PDF 基本資訊與解析工具
# ---------------------------

def get_pdf_page_count(file):
    """回傳 PDF 總頁數。"""
    reader = PdfReader(BytesIO(file.getvalue()))  # 正確用法：不傳 "rb"
    return len(reader.pages)


def extract_page_text(file, page_index):
    """以 pdfminer 擷取單一頁（0-based）的純文字。取不到時回空字串。"""
    try:
        return extract_text(BytesIO(file.getvalue()), page_numbers=[page_index]) or ""
    except Exception:
        return ""


def find_answerkey_page(file):
    """
    從最後頁往前找，回傳第一個包含 'Answer Key'（大小寫不敏感）的頁索引（0-based）。
    找不到則回傳最後一頁。
    """
    n = get_pdf_page_count(file)
    for i in range(n - 1, -1, -1):
        txt = extract_page_text(file, i)
        if re.search(r'(?i)\bAnswer\s*Key', txt):
            return i
    return n - 1


# ---------------------------
# 答案解析
# ---------------------------

def parse_answers_from_text(answer_text):
    """
    從包含答案的文字區段解析出 (題號 -> 選項)。
    支援格式：'1. A'、'1) A'、'1 A'，大小寫不敏感，回傳依題號排序的選項清單。
    """
    # 先抓所有 (題號, 選項)
    # 範例可匹配： "12. B"、"12) b"、"12   a"

    pairs = re.findall(r'(\d{1,3})[\.\)]?\s*([A-Ea-e])', answer_text)
    # print("pairs", pairs)
    if not pairs:
        return []

    # 轉大寫、以題號排序；若同題號重複，取最後出現者
    by_q = {}
    for q, a in pairs:
        by_q[int(q)] = a.upper()

    # 依題號排序
    items = sorted(by_q.items(), key=lambda x: x[0])
    return [a for _, a in items]


def get_answers(files):
    cols = []
    for f in files:
        ak_page = find_answerkey_page(f)
        print("ak_page", ak_page)
        page_text = extract_page_text(f, ak_page) or ""
        print("page_text", page_text)
        # 儘量只拿 'Answer Key' 後半段來解析（大小寫不敏感）
        parts = re.split(r'(?i)\bAnswer\s*Key', page_text, maxsplit=1)
        # print("parts", parts)
        # print("parts0", parts[0])
        # print("parts1", parts[1] if len(parts) == 2 else "")
        answer_text = parts[1] if len(parts) == 2 else page_text
        # print("answer_text", answer_text)
        col = parse_answers_from_text(answer_text)

        if not col:
            st.warning(
                f"在 {f.name} 的第 {ak_page+1} 頁找不到答案樣式。"
                "（可檢查該頁是否為圖片或把頁文字貼上來）"
            )
        cols.append(col)

    max_len = max((len(c) for c in cols), default=0)
    aligned = [c + [""] * (max_len - len(c)) for c in cols]
    return list(zip(*aligned)) if max_len else []



# ---------------------------
# Scramble Map 解析
# ---------------------------

def get_original_question_map(file):
    """
    解析最後一頁的 Scramble Map（右欄是原始題號）。
    預期格式為每行 '顯示題號  原始題號'（空白分隔），例如：
        1  12
        2  5
        3  36
    回傳以顯示題號為索引的 DataFrame（欄名為檔名中的 Ver_x）。
    """
    total_pages = get_pdf_page_count(file)
    ver = re.findall(r"Ver_(\d+)", file.name)
    ver_tag = ver[0] if ver else "X"

    with pdfplumber.open(BytesIO(file.getvalue())) as pdf:
        page = pdf.pages[total_pages - 1]  # 多數 Cognero 試卷 Map 在最後一頁
        text = page.extract_text() or ""

    # 允許行首空白、多個空白/tab 分隔
    # 例如 "\n  12    36" or "\n12\t36"
    pairs = re.findall(r'\n\s*(\d{1,3})\s+(\d{1,3})', text)

    df = pd.DataFrame(pairs, columns=["question", str(ver_tag)])
    if df.empty:
        # 若最後一頁沒有配對，試著在全檔文字搜一輪（保險）
        whole = ""
        for i in range(total_pages):
            whole += extract_page_text(file, i) + "\n"
        pairs = re.findall(r'\n\s*(\d{1,3})\s+(\d{1,3})', whole)
        df = pd.DataFrame(pairs, columns=["question", str(ver_tag)])

    if df.empty:
        # 還是空：回傳空表（避免後續爆炸）
        return pd.DataFrame(columns=[str(ver_tag)]).astype(int)

    df["question"] = df["question"].astype(int)
    df[str(ver_tag)] = df[str(ver_tag)].astype(int)
    df.set_index("question", inplace=True)
    return df.sort_index()


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Extract Correct Answers (Cognero)", layout="wide")

st.sidebar.subheader("上傳試卷 PDF 檔")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF file(s)", accept_multiple_files=True, key="pdf_uploader_v1"
)

files = sorted(uploaded_files, key=lambda x: x.name) if uploaded_files else []

st.subheader("此功能僅限於 Macmillan (Cognero) Test Generator 之試卷")

if files and st.button("Extract Answers"):
    # 解析答案
    answers_rows = get_answers(files)
    answer_df = pd.DataFrame(answers_rows)

    # 設定題號索引從 1 開始
    if not answer_df.empty:
        answer_df.index = answer_df.index + 1
        # 欄名以版本順序（1..N），或你也可改成檔名
        answer_df.columns = [i + 1 for i in range(answer_df.shape[1])]

    # 解析 Scramble Map
    if len(files) == 1:
        map_df = get_original_question_map(files[0])
    else:
        map_list = [get_original_question_map(f) for f in files]
        # 以顯示題號對齊、多欄合併（每欄是各版本原始題號）
        map_df = pd.concat(map_list, axis=1).sort_index()

    col1, col2 = st.columns(2)
    col1.subheader("Correct Answers")
    col1.dataframe(answer_df, width='stretch', height=480)

    col2.subheader("Scramble Map")
    col2.dataframe(map_df, width='stretch', height=480)

    st.write("下載前請檢查試卷版本與答案是否相符。")
    st.write("如有送分題，可在 Excel 中全選答案儲存格後以 Conditional Formatting 加入公式，例如：")
    st.code('=ROW()=MATCH("36",map!B:B,0)   // 其中 36 為範例原始題號', language="text")
    st.write("將要送分的答案改為 A / B / C / D / E。")

    # 產製 Excel（answers 與 map 兩張工作表）
    try:
        xls_bytes = into_excel(answers=answer_df, map=map_df)
        st.download_button(
            label="Download Excel File",
            data=xls_bytes,
            file_name="correct_answers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"匯出 Excel 失敗：{e}")


# 首頁提示
if not files:
    st.info("請先於左側上傳一個或多個 PDF 試卷檔，再點擊 **Extract Answers**。")
