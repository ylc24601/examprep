import pandas as pd
import streamlit as st
from io import StringIO
from master_table import into_excel


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


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# Parameter Setting
st.subheader("參數設定")
col1, col2, col3 = st.columns(3)

qnum = col1.number_input(label="題數", value=50)
point = col2.number_input(label="每題分數", value=2.0)
col3.metric(label="總分", value = qnum * point)
from_cognero = col1.checkbox(label="從Macmillan網站出題", value=True)
if qnum * point != 100:
    st.warning("注意: 總分不是100分!")
st.write("---")
# Sidebar Setting
st.sidebar.subheader("上傳 masterTable.xlsx")
uploaded_mt = st.sidebar.file_uploader("檔案格式: xlsx", key=2)
st.sidebar.subheader("上傳教務處提供之學生作答檔案")
student_answer_file = st.sidebar.file_uploader(
    "Upload txt file", accept_multiple_files=False, key=4)
st.sidebar.subheader("上傳正確答案")
correct_answers = st.sidebar.file_uploader("檔案格式: xlsx", key=5)

if uploaded_mt is not None:
    df = pd.read_excel(uploaded_mt, index_col=0)
    version_num = df["Version"].nunique()
    st.subheader("Master Table")
    st.dataframe(df)
if student_answer_file is not None:
    st_ans_df, problem_df = answer_dataframe(student_answer_file, qnum)
    st.subheader("讀卡結果")
    st.dataframe(st_ans_df)
    st.subheader("異常情形")
    st.dataframe(problem_df)
if correct_answers is not None:
    ca_df = pd.read_excel(correct_answers, index_col=0).T
    ca_df = ca_df.apply(lambda x: x.str.upper())
    st.subheader("正確答案")
    st.dataframe(ca_df)
    xl = pd.ExcelFile(correct_answers)
    sheets_num = len(xl.sheet_names)
    if sheets_num == 2:
        scramble_map = pd.read_excel(
            correct_answers, index_col=0, sheet_name=1, header=0)
        st.subheader("版本題目對照表")
        st.dataframe(scramble_map)
col1, col2, col3, col4, col5 = st.columns(5)
if uploaded_mt is not None:
    col1.metric(label="Master Table人數", value=len(df))
    col4.metric(label="Master Table版本數", value=df.Version.nunique())
if student_answer_file is not None:
    col2.metric(label="讀卡總人數", value=len(st_ans_df))
    col3.metric(label="讀卡異常人數", value=len(problem_df))
if correct_answers is not None:
    col5.metric(label="正確答案版本數", value=len(ca_df))
if student_answer_file and correct_answers and uploaded_mt is not None:
    if df.Version.nunique() != len(ca_df):
        st.warning("注意! 版本數不符")
    if len(df) != len(st_ans_df):
        st.warning("注意! 作答人數與讀卡人數不符")

    if st.button(label="Calculate"):
        if not from_cognero:
            scramble_map = None
        results = grade_cal(st_ans_df, df, ca_df, from_cognero,
                            qnum=qnum, point=point, scramble_map=scramble_map)
        result_df = pd.DataFrame(results[0], columns=('ID', 'Score'))
        detail_df = pd.DataFrame(results[1])
        col1, col2 = st.columns(2)
        col1.subheader("學生成績")
        col1.dataframe(result_df)
        if len(results) == 3:
            correctness_df = pd.DataFrame(
                results[2], index=range(1, qnum+1), columns=("correct_num",))
            correctness_df["Percent"] = correctness_df['correct_num'] * \
                100/len(result_df)
            correctness_df = correctness_df.round(1)
            col2.subheader("試題答對率")
            col2.dataframe(correctness_df)
        st.subheader("學生個別答題情形")
        st.dataframe(detail_df)
        st.subheader("分數統計")
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
        col2.write("上傳至數位學習平台用")