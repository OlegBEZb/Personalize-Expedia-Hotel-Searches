from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def train_test_group_split(*arrays,
                           group_array=None,
                           train_size=None):
    """


    Example
    X_train2, X_test2, y_train2, y_test2, groups_train2, groups_test2 = train_test_group_split(X, y, groups, group_array=groups, train_size=0.9)
    """

    grp_changes = group_array.shift() != group_array
    grp_changes_train_approx = grp_changes.iloc[:int(len(grp_changes) * train_size)]
    split_index = grp_changes_train_approx[grp_changes_train_approx].last_valid_index()
    return train_test_split(*arrays, train_size=split_index, shuffle=False)


def get_time_features(pdf: pd.DataFrame, time_col):
    pdf['day'] = pdf[time_col].dt.day.astype(np.int8)
    pdf['month'] = pdf[time_col].dt.month.astype(np.int8)
    pdf['year'] = pdf[time_col].dt.year.astype(np.int16)
    pdf['quarter'] = pdf[time_col].dt.quarter.astype(np.int8)
    pdf['week'] = pdf[time_col].dt.isocalendar().week.astype(np.int8)
    pdf['dow'] = (pdf[time_col].dt.dayofweek + 1).astype(np.int8)
    pdf['doy'] = pdf[time_col].dt.dayofyear.astype(np.int16)
    pdf['days_in_month'] = pdf[time_col].dt.days_in_month.astype(np.int8)

    pdf["is_weekend"] = pdf["dow"].apply(lambda x: 1 if x in [6, 7] else 0).astype(np.int8)
    pdf['is_month_start'] = pdf[time_col].dt.is_month_start.astype(np.int8)
    pdf['is_month_end'] = pdf[time_col].dt.is_month_end.astype(np.int8)
    pdf['is_quarter_start'] = pdf[time_col].dt.is_quarter_start.astype(np.int8)
    pdf['is_quarter_end'] = pdf[time_col].dt.is_quarter_end.astype(np.int8)
    pdf['is_year_start'] = pdf[time_col].dt.is_year_start.astype(np.int8)
    pdf['is_year_end'] = pdf[time_col].dt.is_year_end.astype(np.int8)

    print('basics are calculated')

    def get_week_id(a_year, a_month, a_week):
        y = a_year
        if a_month == 12 and a_week == 1:
            y = a_year + 1
        elif a_month == 1 and a_week > 50:
            y = a_year - 1
        return y * 100 + a_week

    pdf['week_id'] = pdf.apply(lambda row: get_week_id(row.year, row.month, row.week), axis=1).astype(np.int32)
    # pdf['week_start'] = pdf.groupby(['week_id'])[time_col].transform('min')
    # pdf['week_end'] = pdf.groupby(['week_id'])[time_col].transform('max')
    pdf["season_num"] = (((pdf["month"]) // 3) % 4 + 1).astype(np.int8)
    pdf["week_summer_index"] = pdf["week"].apply(
        lambda w: w - 3 if w >= 3 and w <= 28 else 54 - w if w >= 29 else 0).astype(np.int8)

    print('week-related are calculated')

    return pdf
