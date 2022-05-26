import gc
from copy import deepcopy

import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def get_time_features(pdf: pd.DataFrame, time_col, prefix=None, within_hour_features=True, hour_features=True,
                      day_features=True, week_features=True, month_features=True, season_features=True,
                      year_features=True):
    if prefix is None:
        prefix = time_col + '_'

    if within_hour_features:
        pdf[prefix + 'minute'] = pdf[time_col].dt.minute.astype(np.int8)

    if hour_features:
        pdf[prefix + 'hour'] = pdf[time_col].dt.hour.astype(np.int8)
        pdf[prefix + 'morning'] = (pdf[prefix + 'hour'] >= 7) & (pdf[prefix + 'hour'] < 12)
        pdf[prefix + 'afternoon'] = (pdf[prefix + 'hour'] >= 12) & (pdf[prefix + 'hour'] < 18)
        pdf[prefix + 'evening'] = (pdf[prefix + 'hour'] >= 18) & (pdf[prefix + 'hour'] < 23)
        pdf[prefix + "late_evening"] = (pdf[prefix + 'hour'] == 23) | (pdf[prefix + 'hour'] < 2)
        pdf[prefix + "night"] = (pdf[prefix + 'hour'] >= 2) & (pdf[prefix + 'hour'] < 7)
        pdf[prefix + "work_hour"] = (pdf[prefix + 'hour'] >= 8) & (pdf[prefix + 'hour'] < 18)
        pdf[prefix + "lunch"] = (pdf[prefix + 'hour'] >= 11) & (pdf[prefix + 'hour'] <= 13)

    if day_features:
        pdf[prefix + 'day'] = pdf[time_col].dt.day.astype(np.int8)
        pdf[prefix + 'is_month_start'] = pdf[time_col].dt.is_month_start.astype(np.int8)
        pdf[prefix + 'is_month_end'] = pdf[time_col].dt.is_month_end.astype(np.int8)
        pdf[prefix + 'dow'] = (pdf[time_col].dt.dayofweek + 1).astype(np.int8)
        pdf[prefix + 'is_weekend'] = pdf[prefix + 'dow'].parallel_apply(lambda x: 1 if x in [6, 7] else 0).astype(
            np.int8)
        pdf[prefix + 'doy'] = pdf[time_col].dt.dayofyear.astype(np.int16)

    if week_features:
        pdf[prefix + 'week'] = pdf[time_col].dt.isocalendar().week.astype(np.int8)
        pdf[prefix + 'week_mid_summer_index'] = pdf[prefix + 'week'].parallel_apply(
            lambda w: w - 3 if w >= 3 and w <= 28 else 54 - w if w >= 29 else 0).astype(np.int8)

    if month_features:
        pdf[prefix + 'month'] = pdf[time_col].dt.month.astype(np.int8)
        pdf[prefix + 'days_in_month'] = pdf[time_col].dt.days_in_month.astype(np.int8)

    if season_features:
        pdf[prefix + 'quarter'] = pdf[time_col].dt.quarter.astype(np.int8)
        # pdf[prefix + 'is_quarter_start'] = pdf[time_col].dt.is_quarter_start.astype(np.int8)
        # pdf[prefix + 'is_quarter_end'] = pdf[time_col].dt.is_quarter_end.astype(np.int8)
        pdf[prefix + 'season_num'] = (((pdf[prefix + 'month']) // 3) % 4 + 1).astype(np.int8)

    if year_features:
        pdf[prefix + 'year'] = pdf[time_col].dt.year.astype(np.int16)
        # pdf[prefix + 'is_year_start'] = pdf[time_col].dt.is_year_start.astype(np.int8)
        # pdf[prefix + 'is_year_end'] = pdf[time_col].dt.is_year_end.astype(np.int8)

    def get_week_id(a_year, a_month, a_week):
        y = a_year
        if a_month == 12 and a_week == 1:
            y = a_year + 1
        elif a_month == 1 and a_week > 50:
            y = a_year - 1
        return y * 100 + a_week

    if week_features and month_features and year_features:
        pdf[prefix + 'week_id'] = pdf.parallel_apply(lambda row: get_week_id(row[prefix + 'year'],
                                                                             row[prefix + 'month'],
                                                                             row[prefix + 'week']), axis=1).astype(
            np.int32)

    return pdf


def num_transformations(df, cols, powers=[0.33, 0.5, 2, 3], log_bases=[2, 10, np.e],
                        do_reciprocal=True, do_exp=True):
    for c in cols:
        for p in powers:
            if (p <= 2) or (p > 2 and all(df[c] < 100)):
                df[f'{c}_pow_{p}'] = df[c] ** p

        for log_base in log_bases:
            df[f'{c}_log_{log_base}'] = np.log(df[c] + 1e-6) / np.log(log_base)

        if do_reciprocal:
            df[c + '_reciprocal'] = 1 / (df[c] + 1e-6)

        if do_exp and all(df[c] < 10):
            df[c + '_exp'] = np.exp(df[c])
    return df


def normalize_features(input_df: pd.DataFrame, group_key, target_column, take_log10=False):
    input_df = input_df.copy(deep=True)
    # for numerical stability
    if take_log10:
        input_df[target_column + '_log10'] = np.log10(input_df[target_column] + 1e-2)
        target_column += '_log10'

    aggregations = ["mean", "std"]
    df = input_df.groupby(group_key)[target_column].agg(aggregations)
    col = {}
    for agg in aggregations:
        col[agg] = target_column + "_" + agg
    df.rename(columns=col, inplace=True)

    df_merge = input_df.merge(df.reset_index(), on=group_key)

    suffix = "_norm_by_" + group_key
    df_merge[target_column + suffix] = (df_merge[target_column] -
                                        df_merge[target_column + "_mean"]) \
                                       / df_merge[target_column + "_std"]
    cols_to_drop = list(col.values())
    if take_log10:
        cols_to_drop.append(target_column)
    df_merge = df_merge.drop(labels=cols_to_drop, axis=1)

    del input_df
    gc.collect()
    return df_merge


def create_composite_features(input_df: pd.DataFrame, groupby_cols, feature_value_col: str,
                              agg_methods=["mean", "median", "min", "max"],
                              relative_diff=True,
                              comparison_cols=[], ):
    input_df = input_df.copy(deep=True)

    grp = input_df.groupby(groupby_cols)
    print(f'{grp.ngroups} groups when aggregation by {groupby_cols}')
    new_feature_df = grp[feature_value_col].agg(agg_methods)

    name_mapping = {meth: meth + '_' + feature_value_col + '_per_' + '_per_'.join(groupby_cols)
                    for meth in agg_methods}
    new_feature_df.rename(columns=name_mapping, inplace=True)

    input_df = input_df.join(new_feature_df, on=groupby_cols, how='left')

    for new_feature_name in [c for c in list(name_mapping.values()) if not c.startswith('sum_')]:
        input_df[feature_value_col + '_diff_to_' + new_feature_name] = input_df[feature_value_col] - input_df[
            new_feature_name]
        if relative_diff:
            input_df[feature_value_col + '_rel_diff_to_' + new_feature_name] = (input_df[feature_value_col] - input_df[
                new_feature_name]) / (input_df[feature_value_col] + 1e-2)

        for comparison_col in [c for c in input_df.columns if c in comparison_cols]:
            input_df[comparison_col + '_diff_to_' + new_feature_name] = input_df[comparison_col] - input_df[
                new_feature_name]
            if relative_diff:
                input_df[comparison_col + '_rel_diff_to_' + new_feature_name] = (input_df[comparison_col] - input_df[
                    new_feature_name]) / (input_df[feature_value_col] + 1e-2)

    gc.collect()
    return input_df


# optimize with masks?
def apply_composite_for_splits(df, split_cols=['train', 'val', 'test', 'subm'],
                               **composite_kwargs):
    if split_cols is None:
        df = create_composite_features(df, **composite_kwargs)
    else:
        df_list = []
        for split_col in split_cols:
            split_df_enriched = create_composite_features(df[df[split_col]], **composite_kwargs)
            df_list.append(split_df_enriched)
        df = pd.concat(df_list, axis=0)

    return df
