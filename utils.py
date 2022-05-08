from sklearn.model_selection import train_test_split
from metrics import ndcg
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


def get_target(row):
    """
    0=not clicked at all, 1=clicked but not booked, 5=booked
    """
    if row.booking_bool > 0:
        return 5
    if row.click_bool > 0:
        return 1
    return 0


def predict_in_format(model, data, pool, group_col, predict_item_col, gt_col=None):
    preds = model.predict(pool)

    values = {group_col: data[group_col],
              predict_item_col: data[predict_item_col],
              'pred': preds}

    values_df = pd.DataFrame(values)
    values_df.sort_values(by=[group_col, 'pred'], ascending=[True, False], inplace=True)

    if gt_col is not None:
        values_df['gt'] = gt_col
        ndcg_score = values_df.groupby(group_col)['gt'].apply(ndcg, at=5).mean()
        print('Local test NDCG@5:', ndcg_score)

    return values_df
