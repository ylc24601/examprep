import pandas as pd
import streamlit as st
from io import StringIO
from master_table import into_excel  # reuse existing Excel export helper

def parse_student_answers(student_answer_file, question_num):
    data = StringIO(student_answer_file.read().decode("utf-8"))
    rows = []
    for line in data.readlines():
        sid = line[7:16]
        ans_str = line[16:question_num+16]
        answers = list(ans_str)
        rows.append([sid] + answers)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ID"] + [f"Q{i}" for i in range(1, question_num+1)])
    df.dropna(inplace=True)
    df.rename(columns={0: 'ID'}, inplace=True)
    df.set_index('ID', inplace=True)
    problem_df = df[~df.isin(['A','B','C','D','E']).all(axis=1)]
    df.reset_index(inplace=True)
    df.columns = ["ID"] + [f"Q{i}" for i in range(1, question_num+1)]
    return df, problem_df

def split_indices(q_bio, q_mol):
    bio_idx = list(range(1, q_bio+1))
    mol_idx = list(range(q_bio+1, q_bio+q_mol+1))
    return bio_idx, mol_idx

def grade_weighted(st_ans_df, mt_df, ca_df, w_bio, w_mol, bio_idx, mol_idx, from_cognero=True, scramble_map=None):
    k = len(bio_idx) + len(mol_idx)
    ans_array = st_ans_df.to_numpy()
    results = []
    detail_rows = []
    correct_counts = [0]*k

    mt_df = mt_df.copy()
    mt_df['ID'] = mt_df['ID'].astype(str)

    progress = st.progress(0)
    for idx, row in enumerate(ans_array):
        sid = row[0]
        ver_vals = mt_df.loc[mt_df['ID'] == sid, 'Version'].values
        ver = str(ver_vals[0]) if len(ver_vals) else None
        per_q_marks = []
        score_bio = 0.0
        score_mol = 0.0
        for q in range(1, k+1):
            ans = str(row[q]).upper() if pd.notna(row[q]) else ''
            correct_cell = ca_df.loc[ver, q] if ver is not None else ''
            is_correct = ans in str(correct_cell)
            if is_correct:
                if q in bio_idx:
                    score_bio += w_bio
                elif q in mol_idx:
                    score_mol += w_mol
                correct_counts[q-1] += 1
                per_q_marks.append('O')
            else:
                per_q_marks.append('X')
        results.append([sid, score_bio, score_mol])
        detail_rows.append([sid] + per_q_marks)
        progress.progress((idx+1)/len(ans_array))

    result_df = pd.DataFrame(results, columns=['ID','Biochem_Score','MolBio_Score'])
    detail_df = pd.DataFrame(detail_rows, columns=['ID'] + [f"Q{i}" for i in range(1, k+1)])
    correctness_df = pd.DataFrame({
        'Question': list(range(1, k+1)),
        'Correct_Num': correct_counts
    }).set_index('Question')
    correctness_df['Percent'] = (correctness_df['Correct_Num']*100/len(result_df)).round(1)
    return result_df, detail_df, correctness_df

st.title("雙科成績計算")

st.subheader("題數與配分設定")
# 使用者可自訂科目名稱（預設為生物化學、分子生物學）
name_col1, name_col2 = st.columns(2)
subj_bio = name_col1.text_input("科目A名稱", value="生物化學")
subj_mol = name_col2.text_input("科目B名稱", value="分子生物學")

colA, colB, colC, colD = st.columns(4)
q_bio = colA.number_input(f"{subj_bio}題數", min_value=1, max_value=200, value=10)
w_bio = colB.number_input(f"{subj_bio}每題分值", min_value=0.0, step=0.5, value=2.0)
q_mol = colC.number_input(f"{subj_mol}題數", min_value=1, max_value=200, value=5)
w_mol = colD.number_input(f"{subj_mol}每題分值", min_value=0.0, step=0.5, value=2.0)
q_total = q_bio + q_mol
bio_total = q_bio * w_bio
mol_total = q_mol * w_mol
m1, m2 = st.columns(2)
m1.metric(f"{subj_bio}小計", bio_total)
m2.metric(f"{subj_mol}小計", mol_total)

bio_idx, mol_idx = split_indices(q_bio, q_mol)

st.write("---")

st.sidebar.subheader("上傳 masterTable.xlsx")
uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key=100)

st.sidebar.subheader("上傳教務處提供之學生作答檔案 (TXT)")
student_answer_file = st.sidebar.file_uploader("Upload txt file", type=["txt"], key=101)

st.sidebar.subheader("上傳正確答案 (Excel)，第一個工作表為各版答案，選用者第二表為版本題目對照")
correct_answers = st.sidebar.file_uploader("檔案格式: xlsx", key=102)

mt_df = None
st_ans_df = None
ca_df = None
scramble_map = None

if uploaded_mt is not None:
    mt_df = pd.read_excel(uploaded_mt, index_col=0)
    st.subheader("Master Table")
    st.dataframe(mt_df)

if student_answer_file is not None:
    st_ans_df, problem_df = parse_student_answers(student_answer_file, q_total)
    st.subheader("讀卡結果")
    # 使用顏色標註不同科目欄位
    bio_cols = ['ID'] + [f"Q{i}" for i in range(1, q_bio+1)]
    mol_cols = ['ID'] + [f"Q{i}" for i in range(q_bio+1, q_total+1)]
    st.dataframe(st_ans_df.style.set_properties(subset=bio_cols, **{'background-color': '#e0f7fa'}).set_properties(subset=mol_cols, **{'background-color': '#fce4ec'}))

    # === 分科檢視：確認 split_indices 是否正確 ===
    bio_cols = ['ID'] + [f"Q{i}" for i in range(1, q_bio+1)]
    mol_cols = ['ID'] + [f"Q{i}" for i in range(q_bio+1, q_total+1)]

    col_bio, col_mol = st.columns(2)
    with col_bio:
        st.markdown(f"**{subj_bio}（Q1–Q{q_bio}）**")
        st.dataframe(st_ans_df[bio_cols].style.set_properties(subset=bio_cols[1:], **{'background-color': '#e0f7fa'}))
    with col_mol:
        st.markdown(f"**{subj_mol}（Q{q_bio+1}–Q{q_total}）**")
        st.dataframe(st_ans_df[mol_cols].style.set_properties(subset=mol_cols[1:], **{'background-color': '#fce4ec'}))

    st.subheader("異常情形 (非 A–E)")
    st.dataframe(problem_df)

if correct_answers is not None:
    ca_df = pd.read_excel(correct_answers, index_col=0).T
    ca_df = ca_df.apply(lambda x: x.str.upper())
    # 確保用版本號索引時型別一致（下方以字串索引）
    ca_df.index = ca_df.index.astype(str)
    st.subheader("正確答案表")
    st.dataframe(ca_df)
    xl = pd.ExcelFile(correct_answers)
    if len(xl.sheet_names) == 2:
        scramble_map = pd.read_excel(correct_answers, index_col=0, sheet_name=1, header=0)
        st.subheader("版本題目對照表 (Scramble Map)")
        st.dataframe(scramble_map)

col1, col2, col3, col4 = st.columns(4)
if mt_df is not None:
    col1.metric("Master Table人數", len(mt_df))
    col2.metric("Master Table版本數", mt_df['Version'].nunique())
if st_ans_df is not None:
    col3.metric("讀卡總人數", len(st_ans_df))
if ca_df is not None:
    col4.metric("正確答案版本數", len(ca_df))

if st_ans_df is not None and len(st_ans_df.columns) != (q_total + 1):
    st.warning("讀卡檔題數與設定的總題數不一致，請確認！")

if mt_df is not None and ca_df is not None and mt_df['Version'].nunique() != len(ca_df):
    st.warning("Master Table 的版本數與『正確答案』版本數不一致！")

st.write("---")

if st.button("開始計分 / Analyze"):
    if (mt_df is None) or (st_ans_df is None) or (ca_df is None):
        st.error("請先上傳 Master Table、讀卡檔與正確答案！")
    else:
        result_df, detail_df, correctness_df = grade_weighted(
            st_ans_df, mt_df, ca_df, w_bio, w_mol, bio_idx, mol_idx,
            from_cognero=True, scramble_map=scramble_map
        )
        # 動態科目名稱欄位調整
        bio_col = f"{subj_bio}_Score"
        mol_col = f"{subj_mol}_Score"
        result_df = result_df.rename(columns={
            'Biochem_Score': bio_col,
            'MolBio_Score': mol_col
        })
        colL, colR = st.columns(2)
        colL.subheader("學生成績 (分科)")
        colL.dataframe(result_df)
        colR.subheader("每題答對率 (以施測順序統計)")
        colR.dataframe(correctness_df)

        st.subheader("學生個別答題 O/X")
        st.dataframe(detail_df)

        st.subheader("分數統計")
        st.write(result_df[[bio_col, mol_col]].describe().T)
        
        score_xls = into_excel(
            score=result_df,
            detail=detail_df,
            correctness=correctness_df,
            stats=result_df[[bio_col, mol_col]].describe()
        )
        csv_total = result_df.to_csv(index=False).encode('utf-8')
        col1, col2 = st.columns(2)
        col1.download_button(label='Download Excel (Score + Detail + Item stats)', data=score_xls, file_name='score.xlsx')
        col2.download_button(label='Download CSV (Scores)', data=csv_total, file_name='score.csv', mime='text/csv')
