import catboost
from catboost import CatBoostRanker, Pool, MetricVisualizer, cv
import pandas as pd

pd.set_option('max_columns', None)
import os
import gc

data_path = 'data'

cols_to_use = []

CAT_FEATURES = ['srch_id',
                'site_id',
                'visitor_location_country_id',
                'prop_country_id',
                'prop_id',
                'srch_destination_id',
                'week_id',
                'season_num', 'day', 'month', 'year', 'quarter', 'week', 'dow'
                ]

CAT_FEATURES = [c for c in CAT_FEATURES if c in cols_to_use]

group_col = 'srch_id'
predict_item_col = 'prop_id'

X_train = pd.read_feather(os.path.join(data_path, 'X_train.feather'), columns=cols_to_use)
y_train = pd.read_feather(os.path.join(data_path, 'y_train.feather'))['target']
print('X_train.shape', X_train.shape)

train_pool = Pool(data=X_train,
                  label=y_train,
                  group_id=X_train[group_col],
                  cat_features=CAT_FEATURES,
                  )
del X_train, y_train;
gc.collect()

X_val = pd.read_feather(os.path.join(data_path, 'X_val.feather'), columns=cols_to_use)
y_val = pd.read_feather(os.path.join(data_path, 'y_val.feather'))['target']
print('X_val.shape', X_val.shape)

val_pool = Pool(data=X_val,
                label=y_val,
                group_id=X_val[group_col],
                cat_features=CAT_FEATURES,
                )

del X_val, y_val;
gc.collect()

params = {
    "iterations": 500,
    'loss_function': 'YetiRankPairwise',  # YetiRank should be faster # hints=skip_train~false
    #     'custom_metric': ['NDCG:top=5;type=Base;denominator=LogPosition;hints=skip_train~false'], # :
    "verbose": False,
    'early_stopping_rounds': 400,
    'use_best_model': True,
    #     'metric_period': 50,
    "task_type": "GPU",
}

model = CatBoostRanker(**params)
model.fit(train_pool, eval_set=val_pool, plot=False, verbose_eval=True)
model.save_model('catboost_model')
