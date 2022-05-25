import os
import time
import gc
import json

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd
from skopt import dump

import matplotlib.pyplot as plt

import shap

from features_dict import features, CAT_FEATURES
from utils import flatten_list
from utils import prepare_cats

################## PARAMS START ##################

DATA_PATH = './data'  # should be created during data processing
OUTPUT_FOLDER = './outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cols_to_use = flatten_list(features.values())

CAT_FEATURES = [c for c in CAT_FEATURES if c in cols_to_use]

GROUP_COL = 'srch_id'
PREDICT_ITEM_COL = 'prop_id'

TASK_TYPE = 'GPU'

FIT_MODEL_NOT_LOAD = True
TUNE_MODEL = True
TOTAL_OPTIMIZE_STEPS = 10
INITIAL_RANDOM_OPTIMIZE_STEPS = 4
TUNING_BOOSTING_ITERATIONS = 5000
REGULAR_BOOSTING_ITERATIONS = 6000

################## PARAMS END ##################
################## DATA START ##################

X_train = pd.read_feather(os.path.join(DATA_PATH, 'X_train.feather'), columns=cols_to_use)
prepare_cats(X_train, CAT_FEATURES)
y_train = pd.read_feather(os.path.join(DATA_PATH, 'y_train.feather'))['target']
print('X_train.shape', X_train.shape)

train_pool = Pool(data=X_train,
                  label=y_train,
                  group_id=X_train[GROUP_COL],
                  cat_features=CAT_FEATURES,
                  )

X_val = pd.read_feather(os.path.join(DATA_PATH, 'X_val.feather'), columns=cols_to_use)
prepare_cats(X_val, CAT_FEATURES)
y_val = pd.read_feather(os.path.join(DATA_PATH, 'y_val.feather'))['target']
print('X_val.shape', X_val.shape)

val_pool = Pool(data=X_val,
                label=y_val,
                group_id=X_val[GROUP_COL],
                cat_features=CAT_FEATURES,
                )

X_test = pd.read_feather(os.path.join(DATA_PATH, 'X_test.feather'), columns=cols_to_use)
prepare_cats(X_test, CAT_FEATURES)
y_test = pd.read_feather(os.path.join(DATA_PATH, 'y_test.feather'))['target']
print('X_test.shape', X_test.shape)
test_pool = Pool(data=X_test,
                 label=y_test,
                 group_id=X_test[GROUP_COL],
                 cat_features=CAT_FEATURES,
                 )


################## DATA END ##################


def get_default_model(tuning=False):
    if tuning:
        iterations = TUNING_BOOSTING_ITERATIONS
    else:
        iterations = REGULAR_BOOSTING_ITERATIONS
    model = CatBoostRanker(iterations=iterations,
                           loss_function='YetiRankPairwise',  # YetiRank should be faster # hints=skip_train~false
                           early_stopping_rounds=500,
                           use_best_model=True,
                           task_type=TASK_TYPE,
                           #     'custom_metric': ['NDCG:top=5;type=Base;denominator=LogPosition;hints=skip_train~false'], # :
                           )
    return model


def save_model_params(model_params, path):
    model_params_df = pd.DataFrame.from_dict(model_params, orient='index', columns=['param_value'])
    model_params_df['param_value'] = model_params_df['param_value'].astype('str')
    model_params_df.to_csv(path, index=False)


################## TUNING START ##################

if FIT_MODEL_NOT_LOAD and TUNE_MODEL:
    tuning_start_time = time.time()
    from skopt import gp_minimize
    from skopt.utils import use_named_args
    from skopt.space import Real, Integer, Categorical

    search_space = {
        'depth': Integer(4, 8, prior='uniform', name='depth'),
        'learning_rate': Real(0.01, 0.05, 'uniform', name='learning_rate'),
        'loss_function': Categorical(categories=['YetiRankPairwise', 'YetiRank'], name='loss_function'),
        'nan_mode': Categorical(categories=['Min', 'Max'], name='nan_mode'),
        # On every iteration each possible split gets a score (for example,
        # the score indicates how much adding this split will improve the
        # loss function for the training dataset). The split with the highest
        # score is selected. The scores have no randomness. A normally
        # distributed random variable is added to the score of the feature.
        # It has a zero mean and a variance that decreases during the training.
        # The value of this parameter is the multiplier of the variance.
        'random_strength': Real(1e-2, 20, 'log-uniform', name='random_strength'),
        # too small value makes significant fluctuation
        # 'bagging_temperature': Real(0.0, 5.0, name='bagging_temperature'),
        'border_count': Integer(32, 64, name='border_count'),  # catboost recommends 32, 254
        'l2_leaf_reg': Real(1e-2, 10.0, prior='log-uniform', name='l2_leaf_reg'),
        # too small value makes significant fluctuation
        # 'grow_policy': Categorical(categories=['SymmetricTree', 'Depthwise', 'Lossguide'], name='grow_policy'),
        # Sample rate for bagging.
        # 'subsample': Real(0.1, 1.0, prior='uniform', name='subsample'), for bootstrap_type == "Bernoulli"
        # 'colsample_bylevel': Real(0.3, 1.0, name='colsample_bylevel'),  # aka RMS
        # 'one_hot_max_size': Integer(2, 25, name='one_hot_max_size'),
        # 'langevin': Categorical(categories=[True, False], name='langevin'), # better with True
        # 'boost_from_average': Categorical(categories=[True, False], name='boost_from_average'), FALSE FAILS EVERYTHING
    }


    # this decorator allows your objective function to receive a the parameters as
    # keyword arguments. This is particularly convenient when you want to set
    # scikit-learn estimator parameters
    @use_named_args(list(search_space.values()))
    def objective(**params):
        model = get_default_model(tuning=True)
        print('using params', params)
        model.set_params(**params)

        model.fit(train_pool, eval_set=val_pool, plot=False, verbose_eval=True)

        return -model.eval_metrics(val_pool, 'NDCG:top=5;type=Base;denominator=LogPosition',
                                   ntree_start=model.tree_count_ - 1)['NDCG:top=5;type=Base'][0]


    res_gp = gp_minimize(objective, list(search_space.values()),
                         n_calls=TOTAL_OPTIMIZE_STEPS, n_initial_points=INITIAL_RANDOM_OPTIMIZE_STEPS,
                         initial_point_generator='random',
                         random_state=42, n_jobs=1, verbose=True)

    best_params = {param_name: tuned_param for param_name, tuned_param in zip(search_space.keys(), res_gp.x)}
    print('best_params', best_params)
    # with open(os.path.join(OUTPUT_FOLDER, 'tuned_params.json'), 'w') as fp:
    #     json.dump(best_params, fp)
    save_model_params(best_params, os.path.join(OUTPUT_FOLDER, 'tuned_params_df.csv'))
    try:
        dump(res_gp, os.path.join(OUTPUT_FOLDER, 'skopt_results.pkl'), store_objective=False)
    except:
        pass

    from skopt.plots import plot_convergence, plot_objective, plot_evaluations
    plot_objective(res_gp)
    plt.show()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'objective_plot.jpg'))

    plot_evaluations(res_gp)
    plt.show()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'evaluations_plot.jpg'))

    plot_convergence(res_gp)
    plt.show()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'convergence_plot.jpg'))

################## TUNING END ##################
################## EVAL START ##################

if FIT_MODEL_NOT_LOAD:
    print("Training on train and validating on validation")
    model = get_default_model(tuning=False)
    if TUNE_MODEL:
        print("Using best params from tuned")
        model.set_params(**best_params)
    else:
        print("Using default params")

    model.fit(train_pool, eval_set=val_pool, plot=False, verbose_eval=True)
    model.save_model(os.path.join(OUTPUT_FOLDER, 'catboost_model_train'))

    model_val_params = model.get_all_params()
    model_val_params['tree_count'] = model.tree_count_
    with open(os.path.join(OUTPUT_FOLDER, 'model_params_trained_on_train_stopped_on_val.json'), 'w') as fp:
        json.dump(model_val_params, fp)
else:
    pass
    # model = CatBoostRegressor()
    # model.load_model(model_name, format='cbm')
    # CAT_FEATURES = np.array(model.feature_names_)[model.get_cat_feature_indices()].tolist()

metrics_dict = dict()
metrics_dict['val_NDCG@5'] = model.eval_metrics(val_pool,
                                                'NDCG:top=5;type=Base;denominator=LogPosition',
                                                ntree_start=model.tree_count_ - 1)['NDCG:top=5;type=Base'][0]

metrics_dict['train_NDCG@5'] = model.eval_metrics(train_pool,
                                                  'NDCG:top=5;type=Base;denominator=LogPosition',
                                                  ntree_start=model.tree_count_ - 1)['NDCG:top=5;type=Base'][0]

metrics_dict['test_NDCG@5'] = model.eval_metrics(test_pool,
                                                 'NDCG:top=5;type=Base;denominator=LogPosition',
                                                 ntree_start=model.tree_count_ - 1)['NDCG:top=5;type=Base'][0]

print('eval metrics', metrics_dict)
with open(os.path.join(OUTPUT_FOLDER, 'ndcg_scores_trained_on_train_stopped_on_val.json'), 'w') as fp:
    json.dump(metrics_dict, fp)

################## EVAL END ##################
################## FEATURE IMPORTANCE START ##################


explainer = shap.Explainer(model)
shap_values = explainer(val_pool)  # X_val or val_pool

features = X_val.columns
mean_shaps = np.abs(shap_values.values).mean(0)
shaps_df = pd.DataFrame({'feature': features, 'shap': mean_shaps})
shaps_df.to_csv(os.path.join(OUTPUT_FOLDER, 'shaps_df_trained_on_train_stopped_on_val.csv'), index=False)

################## FEATURE IMPORTANCE END ##################
################## MODEL REFIT START ##################
print("################## MODEL REFIT START ##################")

train_val_pool = Pool(data=pd.concat([X_train, X_val], axis=0),
                      label=pd.concat([y_train, y_val], axis=0),
                      group_id=pd.concat([X_train, X_val], axis=0)[GROUP_COL],
                      cat_features=CAT_FEATURES,
                      )

model = get_default_model(tuning=False)
if TUNE_MODEL:
    print("Using best params from tuned")
    model.set_params(**best_params)
else:
    print("Using default params")

model.fit(train_val_pool, eval_set=test_pool, plot=False, verbose_eval=True)
model.save_model(os.path.join(OUTPUT_FOLDER, 'catboost_model_train_val'))

model_test_params = model.get_all_params()
model_test_params['tree_count'] = model.tree_count_
with open(os.path.join(OUTPUT_FOLDER, 'model_params_trained_on_train_and_val_stopped_on_test.json'), 'w') as fp:
    json.dump(model_test_params, fp)

metrics_dict = dict()
metrics_dict['train_val_NDCG@5'] = model.eval_metrics(train_val_pool,
                                                      'NDCG:top=5;type=Base;denominator=LogPosition',
                                                      ntree_start=model.tree_count_ - 1)['NDCG:top=5;type=Base'][0]

metrics_dict['test_NDCG@5'] = model.eval_metrics(test_pool,
                                                 'NDCG:top=5;type=Base;denominator=LogPosition',
                                                 ntree_start=model.tree_count_ - 1)['NDCG:top=5;type=Base'][0]

print('test metrics', metrics_dict)
with open(os.path.join(OUTPUT_FOLDER, 'ndcg_scores_trained_on_train_and_val_stopped_on_test.json'), 'w') as fp:
    json.dump(metrics_dict, fp)
################## MODEL REFIT END ##################
