import itertools
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from metrics import ndcg


def train_test_group_split(*arrays, group_array=None, train_size=None):
    """


    Example
    X_train2, X_test2, y_train2, y_test2, groups_train2, groups_test2 = train_test_group_split(X, y, groups, group_array=groups, train_size=0.9)
    """

    grp_changes = group_array.shift() != group_array
    grp_changes_train_approx = grp_changes.iloc[: int(len(grp_changes) * train_size)]
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

    values = {
        group_col: data[group_col],
        predict_item_col: data[predict_item_col],
        "pred": preds,
    }

    values_df = pd.DataFrame(values)
    values_df.sort_values(by=[group_col, "pred"], ascending=[True, False], inplace=True)

    if gt_col is not None:
        values_df["gt"] = gt_col
        ndcg_score = values_df.groupby(group_col)["gt"].apply(ndcg, at=5).mean()
        print("Local test NDCG@5:", ndcg_score)

    return values_df


def flatten_list(input_list):
    return list(itertools.chain.from_iterable(input_list))


# use locally or if you added the sample subm file to the dataset
def validate_submission(subm_sample_path, our_subm_path: str, group_col):
    subm_sample = pd.read_csv(subm_sample_path)
    our_subm = pd.read_csv(our_subm_path)

    assert subm_sample.shape == our_subm.shape
    assert our_subm[group_col].equals(subm_sample[group_col])
    assert our_subm.index.equals(subm_sample.index)
    assert (
        subm_sample.groupby(group_col)["prop_id"]
        .apply(len)
        .to_frame()
        .equals(our_subm.groupby(group_col)["prop_id"].apply(len).to_frame())
    )
    print("everything is ok to submit")


def prepare_cats(pd_df: pd.DataFrame, cat_features, cat_fillna_value="NaN_category"):
    cat_fillna_value = "NaN_category"
    for cat_col in cat_features:
        pd_df[cat_col] = pd_df[cat_col].astype("category")
        if cat_fillna_value not in pd_df[cat_col].cat.categories:  # and cat_col not in int2str2cat_cols:
            pd_df[cat_col] = pd_df[cat_col].cat.add_categories(cat_fillna_value)
            pd_df[cat_col] = pd_df[cat_col].fillna(cat_fillna_value)


def downcast(input_df: pd.DataFrame):
    for column in tqdm(input_df.columns, total=input_df.shape[1]):
        if input_df[column].dtype.name.lower() in ["category", "bool"]:
            continue
        elif ("float" in input_df[column].dtype.name.lower()) or (input_df[column].isna().any()):
            input_df[column] = pd.to_numeric(input_df[column].astype(float), downcast="float")
        elif input_df[column].dtype.name.lower().startswith("uint"):
            input_df[column] = pd.to_numeric(input_df[column], downcast="unsigned")
        elif input_df[column].dtype.name.lower().startswith("int"):
            input_df[column] = pd.to_numeric(input_df[column], downcast="integer")
    return input_df
