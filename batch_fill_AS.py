import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ------------------------------------------
# 依需求修正以下參數

FILENAME = '../Output/AnswerSheet.pdf'
COURSE_NAME = '生化-1'

# 以下更換印表機時需微調
ID_LEFT_MARGIN = 20.45
ID_RIGHT_MARGIN = 60.47
ID_TOP_MARGIN = 192.58
ID_BOTTOM_MARGIN = 151.05

# 以下程式碼請勿修改
# ------------------------------------------


def batchFillAS(df_seat, filename, course_name, ID_left, ID_right, ID_top, ID_bottom):

    # Vectorized for answer sheet generation
    df_seat['ID'] = df_seat['ID'].apply(str)
    df_seat_array = df_seat.sort_values(by='Seat_index').to_numpy()
    pdfmetrics.registerFont(TTFont('Microsoft Jhenghei', 'Microsoft Jhenghei.ttf'))
    course = course_name
    # Start a new canvas
    answer_sheet = canvas.Canvas(filename, pagesize=A5)
    # loop through each student in data_array from df
    for ID, Name, Class, Seat_index, Seat, Version in df_seat_array:
        answer_sheet.setFont('Microsoft Jhenghei', 10)
        answer_sheet.drawString(23*mm, 194*mm, ID)
        answer_sheet.drawString(120*mm, 193*mm, Name)
        answer_sheet.drawString(90*mm, 193*mm, Class)
        answer_sheet.drawString(85*mm, 188*mm, course)
        answer_sheet.drawString(112*mm, 188*mm, f'座位：{Seat}')
        # for ID filling
        x = np.linspace(ID_left, ID_right, 10)
        y = np.linspace(ID_top, ID_bottom, 9)
        for index, num in enumerate(ID):
            answer_sheet.rect(x[int(num)]*mm, y[index]*mm, 3.2 * mm, 1 * mm, stroke=1, fill=1)

        # finish editing and save this page
        answer_sheet.showPage()
    answer_sheet.save()


# if __name__ == '__main__':
#     batchFillAS(df, FILENAME, COURSE_NAME, ID_LEFT_MARGIN, ID_TOP_MARGIN, ID_TOP_MARGIN, ID_BOTTOM_MARGIN)
