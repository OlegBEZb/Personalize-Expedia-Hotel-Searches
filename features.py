import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def get_time_features(pdf: pd.DataFrame, time_col):
    pdf['hour'] = pdf[time_col].dt.hour.astype(np.int8)
    pdf['minute'] = pdf[time_col].dt.minute.astype(np.int8)

    pdf['morning_booking'] = ((pdf["hour"] >= 7) & (pdf["hour"] < 12))
    pdf['afternoon_booking'] = ((pdf["hour"] >= 12) & (pdf["hour"] < 18))
    pdf['evening_booking'] = ((pdf["hour"] >= 18) & (pdf["hour"] < 23))
    pdf["late_evening_booking"] = ((pdf["hour"] == 23) | (pdf["hour"] < 2))
    pdf["night_booking"] = ((pdf["hour"] >= 2) & (pdf["hour"] < 7))
    pdf["work_hour_booking"] = ((pdf["hour"] >= 8) & (pdf["hour"] < 18))
    pdf["lunch_booking"] = ((pdf["hour"] >= 11) & (pdf["hour"] <= 13))

    pdf['day'] = pdf[time_col].dt.day.astype(np.int8)
    pdf['month'] = pdf[time_col].dt.month.astype(np.int8)
    pdf['year'] = pdf[time_col].dt.year.astype(np.int16)
    pdf['quarter'] = pdf[time_col].dt.quarter.astype(np.int8)
    pdf['week'] = pdf[time_col].dt.isocalendar().week.astype(np.int8)
    pdf['dow'] = (pdf[time_col].dt.dayofweek + 1).astype(np.int8)
    pdf['doy'] = pdf[time_col].dt.dayofyear.astype(np.int16)
    pdf['days_in_month'] = pdf[time_col].dt.days_in_month.astype(np.int8)

    pdf["is_weekend"] = pdf["dow"].parallel_apply(lambda x: 1 if x in [6, 7] else 0).astype(np.int8)
    pdf['is_month_start'] = pdf[time_col].dt.is_month_start.astype(np.int8)
    pdf['is_month_end'] = pdf[time_col].dt.is_month_end.astype(np.int8)
    pdf['is_quarter_start'] = pdf[time_col].dt.is_quarter_start.astype(np.int8)
    pdf['is_quarter_end'] = pdf[time_col].dt.is_quarter_end.astype(np.int8)
    pdf['is_year_start'] = pdf[time_col].dt.is_year_start.astype(np.int8)
    pdf['is_year_end'] = pdf[time_col].dt.is_year_end.astype(np.int8)

    def get_week_id(a_year, a_month, a_week):
        y = a_year
        if a_month == 12 and a_week == 1:
            y = a_year + 1
        elif a_month == 1 and a_week > 50:
            y = a_year - 1
        return y * 100 + a_week

    pdf['week_id'] = pdf.parallel_apply(lambda row: get_week_id(row.year, row.month, row.week), axis=1).astype(np.int32)
    pdf["season_num"] = (((pdf["month"]) // 3) % 4 + 1).astype(np.int8)
    pdf["week_summer_index"] = pdf["week"].parallel_apply(
        lambda w: w - 3 if w >= 3 and w <= 28 else 54 - w if w >= 29 else 0).astype(np.int8)

    return pdf
