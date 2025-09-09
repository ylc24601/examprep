import pandas as pd
import streamlit as st
from io import StringIO
from master_table import into_excel
import matplotlib.pyplot as plt
import mplfonts
mplfonts.use_font("Noto Sans TC")  # 第一次會下載，之後沿用快取
plt.rcParams["axes.unicode_minus"] = False

# =============================
# Utilities
# =============================

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
        empty_df = pd.DataFrame(columns=["ID"] + [f"Q{i}" for i in range(1, question_num+1)])
        empty_problem = empty_df.copy()
        return empty_df, empty_problem
    df.dropna(inplace=True)
    df.rename(columns={0: 'ID'}, inplace=True)
    df.set_index('ID', inplace=True)
    problem_df = df[~df.isin(['A','B','C','D','E']).all(axis=1)]
    df.reset_index(inplace=True)
    df.columns = ["ID"] + [f"Q{i}" for i in range(1, question_num+1)]
    return df, problem_df


def build_subjects_indices(subjects):
    start = 1
    out = []
    for s in subjects:
        count = int(s["count"]) if pd.notna(s["count"]) else 0
        entry = dict(s)
        entry["start"] = start
        entry["end"] = start + count - 1 if count > 0 else start - 1
        entry["indices"] = list(range(entry["start"], entry["end"] + 1)) if count > 0 else []
        out.append(entry)
        start = entry["end"] + 1
    return out


def sanitize_col_name(name):
    return str(name).replace(' ', '_')


def grade_weighted_multi(st_ans_df, mt_df, ca_df, subjects, show_progress=True):
    k = sum(len(s['indices']) for s in subjects)

    mt_df = mt_df.copy()
    mt_df['ID'] = mt_df['ID'].astype(str)
    id_to_ver = dict(zip(mt_df['ID'], mt_df['Version'].astype(str)))

    # 建立題號對應的科目與分值
    q_to_name, q_to_weight = {}, {}
    for s in subjects:
        w = float(s['weight'])
        for q in s['indices']:
            q_to_name[q] = s['name']
            q_to_weight[q] = w

    correct_counts = [0]*k
    results, detail_rows = [], []
    prog = st.progress(0) if show_progress else None

    ans_array = st_ans_df.to_numpy()

    for idx, row in enumerate(ans_array):
        sid = row[0]
        ver = id_to_ver.get(sid)
        per_subject_score = {s['name']: 0.0 for s in subjects}
        per_q_marks = []

        for q in range(1, k+1):
            ans = str(row[q]).upper() if pd.notna(row[q]) else ''
            correct_cell = ca_df.loc[ver, q] if (ver is not None and q in ca_df.columns) else ''
            if ans in str(correct_cell):
                subj_name = q_to_name.get(q)
                if subj_name is not None:
                    per_subject_score[subj_name] += q_to_weight.get(q, 0.0)
                correct_counts[q-1] += 1
                per_q_marks.append('O')
            else:
                per_q_marks.append('X')

        results.append([sid] + [per_subject_score[s['name']] for s in subjects])
        detail_rows.append([sid] + per_q_marks)
        if show_progress:
            prog.progress((idx+1)/len(ans_array))

    result_cols = ['ID'] + [f"{sanitize_col_name(s['name'])}_Total" for s in subjects]
    result_df  = pd.DataFrame(results, columns=result_cols)
    detail_df  = pd.DataFrame(detail_rows, columns=['ID'] + [f"Q{i}" for i in range(1, k+1)])
    correctness_df = pd.DataFrame({'Question': list(range(1, k+1)), 'Correct_Num': correct_counts}).set_index('Question')
    if len(result_df) > 0:
        correctness_df['Percent'] = (correctness_df['Correct_Num']*100/len(result_df)).round(1)
    else:
        correctness_df['Percent'] = 0.0

    return result_df, detail_df, correctness_df


# =============================
# UI
# =============================

st.title("多科加權成績計算")

st.subheader("題數與配分設定")
if 'subjects_df' not in st.session_state:
    st.session_state.subjects_df = pd.DataFrame([
        {"name": "生物化學", "count": 10, "weight": 10.0, "color": "#e0f7fa"},
        {"name": "分子生物學", "count": 5,  "weight": 20.0, "color": "#fce4ec"},
    ])

subjects_df = st.data_editor(
    st.session_state.subjects_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("科目名稱", required=True),
        "count": st.column_config.NumberColumn("題數", min_value=0, step=1),
        "weight": st.column_config.NumberColumn("每題分值", min_value=0.0, step=0.5),
        "color": st.column_config.TextColumn("顯示顏色 (HEX)")  # 先用文字存 HEX
    },
    hide_index=True,
    key="subjects_editor"
)

# --- 在表格下方提供每列的顏色挑選器，並回寫至 DataFrame ---
edited = subjects_df.copy()
if "color" not in edited.columns:
    edited["color"] = "#ffffff"
edited["color"] = edited["color"].fillna("#ffffff").astype(str)

st.markdown("#### 顏色挑選")
for i, (idx, row) in enumerate(edited.iterrows()):
    c1, c2, c3 = st.columns([2, 3, 3])
    with c1:
        st.write(row.get("name", f"Subject {i+1}") or f"Subject {i+1}")
    with c2:
        picked = st.color_picker(
            "顏色",
            value=row.get("color", "#ffffff") if isinstance(row.get("color", ""), str) else "#ffffff",
            key=f"color_picker_{idx}",
            label_visibility="collapsed"
        )
    edited.at[idx, "color"] = picked

# 存回 session_state，後續你的程式都用 st.session_state.subjects_df
st.session_state.subjects_df = edited
subjects_df = edited

subjects_list = [
    {"name": r.get("name", "Subject"),
     "count": int(r.get("count", 0) or 0),
     "weight": float(r.get("weight", 0.0) or 0.0),
     "color": r.get("color", "#ffffff")}
    for _, r in subjects_df.iterrows() if int(r.get("count", 0) or 0) > 0
]
subjects_list = build_subjects_indices(subjects_list)

q_total = sum(s['count'] for s in subjects_list)
totals = {s['name']: s['count']*s['weight'] for s in subjects_list}

tot_cols = st.columns(min(4, max(2, len(subjects_list))))
for i, (name, subtotal) in enumerate(totals.items()):
    tot_cols[i % len(tot_cols)].metric(f"{name}小計", subtotal)

st.write("---")

# Uploads
st.sidebar.subheader("上傳 masterTable.xlsx")
uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key=200)

st.sidebar.subheader("上傳教務處提供之學生作答檔案 (TXT)")
student_answer_file = st.sidebar.file_uploader("Upload txt file", type=["txt"], key=201)

st.sidebar.subheader("上傳正確答案 (Excel)，第一個工作表為各版答案，第二則為版本題目對照")
correct_answers = st.sidebar.file_uploader("檔案格式: xlsx", key=202)

mt_df = None
st_ans_df = None
ca_df = None
scramble_map = None

if uploaded_mt is not None:
    mt_df = pd.read_excel(uploaded_mt, index_col=0)
    st.subheader("Master Table")
    st.dataframe(mt_df)

if student_answer_file is not None and q_total > 0:
    st_ans_df, problem_df = parse_student_answers(student_answer_file, q_total)
    st.subheader("讀卡結果（分科檢視）")
    tabs = st.tabs([s['name'] for s in subjects_list])
    for tab, s in zip(tabs, subjects_list):
        with tab:
            cols = ['ID'] + [f"Q{i}" for i in s['indices']]
            st.markdown(f"**{s['name']}（Q{s['start']}–Q{s['end']}）**")
            st.dataframe(st_ans_df[cols].style.set_properties(subset=cols[1:], **{'background-color': s['color']}))

    st.subheader("異常情形 (非 A–E)")
    if not problem_df.empty:
        tmp_prob = problem_df.copy().reset_index()
        if 'ID' not in tmp_prob.columns and 'index' in tmp_prob.columns:
            tmp_prob = tmp_prob.rename(columns={'index': 'ID'})

        styled_problem = tmp_prob.style.applymap(
            lambda v: 'background-color: #fff3cd; color: #856404; font-weight: bold'
            if pd.notna(v) and v not in ['A','B','C','D','E'] else ''
        )
        st.dataframe(styled_problem)

        answer_cols = [c for c in tmp_prob.columns if c != 'ID']
        anomaly_counts = tmp_prob[answer_cols].apply(
            lambda row: int(((row.notna()) & (~row.isin(['A','B','C','D','E']))).sum()), axis=1
        )
        anom_df = pd.DataFrame({'ID': tmp_prob['ID'], 'Anomaly_Count': anomaly_counts})
        anom_df = anom_df.sort_values('Anomaly_Count', ascending=False)
        st.subheader("異常次數統計（依學生）")
        st.dataframe(anom_df)
    else:
        st.dataframe(problem_df)

if correct_answers is not None:
    ca_df = pd.read_excel(correct_answers, index_col=0).T
    ca_df = ca_df.apply(lambda x: x.str.upper())
    ca_df.index = ca_df.index.astype(str)
    try:
        ca_df.columns = ca_df.columns.astype(int)
    except Exception:
        pass
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

if student_answer_file is not None and q_total > 0 and len(st_ans_df.columns) != (q_total + 1):
    st.warning("讀卡檔題數與設定的總題數不一致，請確認！")

if mt_df is not None and ca_df is not None and mt_df['Version'].nunique() != len(ca_df):
    st.warning("Master Table 的版本數與『正確答案』版本數不一致！")

st.write("---")

if st.button("開始計分 / Analyze"):
    if (mt_df is None) or (st_ans_df is None) or (ca_df is None) or (q_total == 0):
        st.error("請先上傳 Master Table、讀卡檔、正確答案，並設定題數！")
    else:
        result_df, detail_df, correctness_df = grade_weighted_multi(
            st_ans_df, mt_df, ca_df, subjects_list, show_progress=True
        )
        st.subheader("學生成績（各科總分）")
        st.dataframe(result_df)

        st.subheader("試題分析（難度、鑑別度、選項比例）")
        # 加入難度與鑑別度
        def build_item_analysis(df_ans: pd.DataFrame, correctness_df: pd.DataFrame, result_df: pd.DataFrame, k: int) -> pd.DataFrame:
            # 以總分（各科總分加總）排序，取前後 27% 作為高低分組
            score_cols = [c for c in result_df.columns if c != 'ID']
            tmp = result_df.copy()
            tmp['Total_All'] = tmp[score_cols].sum(axis=1)
            sorted_df = tmp.sort_values('Total_All', ascending=False)
            n = len(sorted_df)
            g = int(n * 0.27) if n > 0 else 0
            top_ids = set(sorted_df.head(g)['ID'])
            bot_ids = set(sorted_df.tail(g)['ID'])

            # 將答案表以 ID 為索引，避免 boolean mask 與索引不對齊（修正你的錯誤）
            ans_idx = df_ans.set_index('ID')

            rows = []
            for i in range(1, k+1):
                correct_total = correctness_df.loc[i, 'Correct_Num'] if i in correctness_df.index else 0
                p = correct_total / n if n > 0 else 0

                col = f"Q{i}"
                top_series = ans_idx.loc[ans_idx.index.isin(top_ids), col].astype(str).str.upper()
                bot_series = ans_idx.loc[ans_idx.index.isin(bot_ids), col].astype(str).str.upper()

                # 取第一個版本當參考正確選項（若無則空）
                try:
                    correct_opt = str(ca_df.iloc[0, i-1])  # 單一選項字元
                except Exception:
                    correct_opt = ''

                p_top = (top_series == correct_opt).mean() if len(top_series) > 0 else 0
                p_bot = (bot_series == correct_opt).mean() if len(bot_series) > 0 else 0
                D = p_top - p_bot

                rows.append({
                    'Question': i,
                    'Correct_Num': correct_total,
                    'Percent': round(p*100, 1) if 'Percent' not in correctness_df.columns else correctness_df.loc[i, 'Percent'],
                    'Difficulty_p': round(p, 3),
                    'Discrimination_D': round(D, 3)
                })
            return pd.DataFrame(rows).set_index('Question')

        item_analysis_df = build_item_analysis(st_ans_df, correctness_df, result_df, len(correctness_df))

        # === 各選項比例 ===
        def build_option_distribution(df_ans: pd.DataFrame, k: int) -> pd.DataFrame:
            rows = []
            for i in range(1, k+1):
                col = f"Q{i}"
                s = df_ans[col].astype(str).str.upper()
                valid = s[s.isin(list('ABCDE'))]
                denom = len(valid)
                pct = {opt: (valid.value_counts().get(opt, 0) * 100.0 / denom if denom > 0 else 0.0) for opt in 'ABCDE'}
                rows.append({
                    'Question': i,
                    'A_%': round(pct['A'], 1),
                    'B_%': round(pct['B'], 1),
                    'C_%': round(pct['C'], 1),
                    'D_%': round(pct['D'], 1),
                    'E_%': round(pct['E'], 1),
                    'N_valid': denom
                })
            return pd.DataFrame(rows).set_index('Question')

        option_pct_df = build_option_distribution(st_ans_df, len(correctness_df))

        # 合併三個分析表
        correctness_full = item_analysis_df.join(option_pct_df)
        # 重新排序與命名欄位
        col_order = ['Correct_Num', 'Percent', 'Difficulty_p', 'Discrimination_D', 'A_%', 'B_%', 'C_%', 'D_%', 'E_%', 'N_valid']
        correctness_full = correctness_full[col_order]
        correctness_full = correctness_full.rename(columns={
            'Percent': 'Correct_%',
            'Difficulty_p': '難度p',
            'Discrimination_D': '鑑別度D'
        })

        # 加入建議判定欄位（難度／鑑別度）
        def _difficulty_label(p: float) -> str:
            if pd.isna(p):
                return ''
            if p < 0.20:
                return '過難'
            if p < 0.40:
                return '偏難'
            if p <= 0.60:
                return '適中'
            if p <= 0.80:
                return '偏易'
            return '過易'
        def _discrimination_label(d: float) -> str:
            if pd.isna(d):
                return ''
            if d < 0:
                return '負向(需檢查)'
            if d < 0.20:
                return '不佳'
            if d < 0.30:
                return '可接受'
            if d < 0.40:
                return '良'
            return '優'
        correctness_full['難度評語'] = correctness_full['難度p'].apply(_difficulty_label)
        correctness_full['鑑別度評語'] = correctness_full['鑑別度D'].apply(_discrimination_label)

        # 調整欄位順序，把評語放在對應指標旁
        final_order = ['Correct_Num', 'Correct_%', '難度p', '難度評語', '鑑別度D', '鑑別度評語', 'A_%', 'B_%', 'C_%', 'D_%', 'E_%', 'N_valid']
        correctness_full = correctness_full[final_order]
        st.dataframe(correctness_full)

        # === 視覺化圖表（可開關） ===
        show_charts = st.checkbox("顯示圖表分析", value=True, key="show_charts")
        if show_charts:
            # 1) 分數分布直方圖（各科）
            score_cols_charts = [c for c in result_df.columns if c != 'ID']
            st.subheader("整體分數分布")
            for col in score_cols_charts:
                fig, ax = plt.subplots()
                ax.hist(result_df[col].dropna(), bins=10, edgecolor='black')
                ax.set_title(f"{col} 分布")
                ax.set_xlabel("分數"); ax.set_ylabel("學生人數")
                st.pyplot(fig)

            # 2) 各科平均分數比較
            st.subheader("各科平均分數比較")
            avg_scores = result_df[score_cols_charts].mean()
            st.bar_chart(avg_scores)

            # 3) 難度 (p 值) 分布
            st.subheader("各題難度 (p 值)")
            fig, ax = plt.subplots()
            ax.bar(correctness_full.index, correctness_full['難度p'])
            ax.axhline(0.4, color='tab:orange', linestyle='--', linewidth=2, label='理想下限 0.4')
            ax.axhline(0.6, color='tab:green',  linestyle='--', linewidth=2, label='理想上限 0.6')
            ax.set_xlabel("題號"); ax.set_ylabel("難度 p")
            ax.legend()
            st.pyplot(fig)

            # 4) 鑑別度 D 值
            st.subheader("各題鑑別度 (D 值)")
            fig, ax = plt.subplots()
            ax.bar(correctness_full.index, correctness_full['鑑別度D'])
            ax.axhline(0.3, color='tab:orange', linestyle='--', label='最低建議 0.3')
            ax.axhline(0.4, color='tab:green', linestyle='--', label='理想值 >0.4')
            ax.set_xlabel("題號"); ax.set_ylabel("鑑別度 D")
            ax.legend()
            st.pyplot(fig)

        # === 每題各選項（A–E）學生選擇比例 ===
        def build_option_distribution(df_ans: pd.DataFrame, k: int) -> pd.DataFrame:
            rows = []
            for i in range(1, k+1):
                col = f"Q{i}"
                s = df_ans[col].astype(str).str.upper()
                valid = s[s.isin(list('ABCDE'))]
                denom = len(valid)
                pct = {opt: (valid.value_counts().get(opt, 0) * 100.0 / denom if denom > 0 else 0.0) for opt in 'ABCDE'}
                rows.append({
                    'Question': i,
                    'A_%': round(pct['A'], 1),
                    'B_%': round(pct['B'], 1),
                    'C_%': round(pct['C'], 1),
                    'D_%': round(pct['D'], 1),
                    'E_%': round(pct['E'], 1),
                    'N_valid': denom
                })
            return pd.DataFrame(rows).set_index('Question')

        option_pct_df = build_option_distribution(st_ans_df, len(correctness_df))
        st.subheader("各選項比例（每題）")
        st.caption("百分比以有效作答（A–E）為分母；N_valid 顯示該題有效作答人數。")
        st.dataframe(option_pct_df)

        st.subheader("學生個別答題 O/X")
        st.dataframe(detail_df)

        score_cols = [c for c in result_df.columns if c != 'ID']
        st.subheader("分數統計")
        st.write(result_df[score_cols].describe().T)

        excel_sheets = {
            'score': result_df,
            'detail': detail_df,
            'correctness': correctness_full,
            'stats': result_df[score_cols].describe(),
            'option_pct': option_pct_df
        }
        score_xls = into_excel(**excel_sheets)
        csv_all = result_df.to_csv(index=False).encode('utf-8')

        colA, colB = st.columns(2)
        colA.download_button(
            label='Download Excel (Score + Detail + Item stats)',
            data=score_xls,
            file_name='MultiSubject_WeightedScore.xlsx'
        )
        colB.download_button(
            label='Download CSV (All scores)',
            data=csv_all,
            file_name='multi_subject_scores.csv',
            mime='text/csv'
        )

        st.subheader("分科成績下載")
        dl_cols = st.columns(min(4, max(2, len(score_cols))))
        for i, s in enumerate(subjects_list):
            col = f"{sanitize_col_name(s['name'])}_Total"
            sub_df = result_df[['ID', col]]
            dl_cols[i % len(dl_cols)].download_button(
                label=f"Download CSV – {s['name']}",
                data=sub_df.to_csv(index=False, header=False).encode('utf-8'),
                file_name=f"{sanitize_col_name(s['name'])}_scores.csv",
                mime='text/csv'
            )
            
