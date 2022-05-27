import os
import time
import json

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from features_dict import features, CAT_FEATURES
from utils import flatten_list, prepare_cats

print('################## PARAMS START ##################')

DATA_PATH = './data_temp'  # should be created during data processing or downloading
OUTPUT_FOLDER = './outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cols_to_use = flatten_list(features.values())

cols_lost_from_v1 = [
    'min_price_per_day_per_prop_country_id',
    'price_per_day_rel_diff_to_min_price_per_day_per_prop_country_id',
    'price_per_day_diff_to_max_price_per_day_per_prop_country_id',
    'price_per_day_diff_to_max_price_per_day_per_visitor_location_country_id_per_prop_review_score',
    'price_per_day_diff_to_max_price_per_day_per_visitor_location_country_id_per_srch_destination_id',
    'price_per_day_rel_diff_to_max_price_per_day_per_visitor_location_country_id_per_srch_destination_id',
    'price_per_day_rel_diff_to_min_price_per_day_per_visitor_location_country_id_per_srch_destination_id_per_prop_starrating',
    'price_per_day_rel_diff_to_median_price_per_day_per_srch_destination_id_per_trip_start_date_quarter_per_prop_review_score'
]

cols_to_use = [c for c in cols_to_use if c not in cols_lost_from_v1]

#### remove me
from random import shuffle
shuffle(cols_to_use)
cols_to_use = cols_to_use[:100]
cols_to_use = list(set(cols_to_use + ['srch_id', 'prop_id', 'random_bool']))
cols_to_use = [c for c in cols_to_use if c not in ['booking_prob_train', 'click_prob_train', 'book_per_click']]  # looks leaky
#### remove above


CAT_FEATURES = [c for c in CAT_FEATURES if c in cols_to_use]

GROUP_COL = 'srch_id'
PREDICT_ITEM_COL = 'prop_id'

TASK_TYPE = 'GPU'

FIT_MODEL_NOT_LOAD = True
TUNE_MODEL = False
TOTAL_OPTIMIZE_STEPS = 3
INITIAL_RANDOM_OPTIMIZE_STEPS = 2
TUNING_BOOSTING_ITERATIONS = 3000
REGULAR_BOOSTING_ITERATIONS = 6000

DO_EVAL = False
DO_REFIT = True

MAKE_PREDS = True

print('################## PARAMS END ##################')
print('################## DATA START ##################')

X_train = pd.read_feather(os.path.join(DATA_PATH, 'X_train.feather'), columns=cols_to_use)
y_train = pd.read_feather(os.path.join(DATA_PATH, 'y_train.feather'))['target']

# ####### remove me
# rand_groups = X_train[X_train['random_bool'] == 1][GROUP_COL].unique()
# from random import shuffle
# shuffle(rand_groups)
# rand_groups = rand_groups[: int(len(rand_groups)/2)]
# X_train = X_train[~X_train[GROUP_COL].isin(rand_groups)]
# y_train = y_train.loc[X_train.index]
# ####### remove above

prepare_cats(X_train, CAT_FEATURES)
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

print('################## DATA END ##################')


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
                           metric_period=50,
                           # custom_metric=['NDCG:top=5;type=Base;denominator=LogPosition;hints=skip_train~false'],
                           )
    return model


def save_model_params(model_params, path):
    model_params_df = pd.DataFrame.from_dict(model_params, orient='index', columns=['param_value'])
    model_params_df['param_value'] = model_params_df['param_value'].astype('str')
    model_params_df.to_csv(path, index=False)


print('################## TUNING START ##################')

if FIT_MODEL_NOT_LOAD and TUNE_MODEL:
    tuning_start_time = time.time()
    from skopt import gp_minimize, dump
    from skopt.utils import use_named_args
    from skopt.space import Real, Integer, Categorical
    from skopt.plots import plot_convergence, plot_objective, plot_evaluations

    search_space = {
        # 'depth': Integer(5, 8, prior='uniform', name='depth'),
        'learning_rate': Real(0.03, 0.2, 'uniform', name='learning_rate'),
        'loss_function': Categorical(categories=['YetiRankPairwise', 'YetiRank'], name='loss_function'),
        'nan_mode': Categorical(categories=['Min', 'Max'], name='nan_mode'),
        # On every iteration each possible split gets a score (for example,
        # the score indicates how much adding this split will improve the
        # loss function for the training dataset). The split with the highest
        # score is selected. The scores have no randomness. A normally
        # distributed random variable is added to the score of the feature.
        # It has a zero mean and a variance that decreases during the training.
        # The value of this parameter is the multiplier of the variance.
        # 'random_strength': Real(1e-2, 20, 'log-uniform', name='random_strength'),
        # too small value makes significant fluctuation
        # 'bagging_temperature': Real(0.0, 5.0, name='bagging_temperature'),
        # 'border_count': Integer(32, 64, name='border_count'),  # catboost recommends 32, 254
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
    # save_model_params(best_params, os.path.join(OUTPUT_FOLDER, 'tuned_params_df.csv'))
    dump(res_gp, os.path.join(OUTPUT_FOLDER, 'skopt_results.pkl'), store_objective=False)

    plot_objective(res_gp)
    plt.show()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'objective_plot.jpg'))

    plt.clf()
    plot_evaluations(res_gp)
    plt.show()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'evaluations_plot.jpg'))

    plt.clf()
    plot_convergence(res_gp)
    plt.show()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'convergence_plot.jpg'))

print('################## TUNING END ##################')
print('################## EVAL START ##################')

if DO_EVAL:

    if FIT_MODEL_NOT_LOAD:
        print("Training on train and validating on validation")
        model = get_default_model(tuning=False)
        if TUNE_MODEL:
            print("Using best params from tuned")
            print(best_params)
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

print('################## EVAL END ##################')
print('################## FEATURE IMPORTANCE START ##################')

explainer = shap.Explainer(model)
shap_values = explainer(val_pool)  # X_val or val_pool

features = X_val.columns
mean_shaps = np.abs(shap_values.values).mean(0)
shaps_df = pd.DataFrame({'feature': features, 'shap': mean_shaps})
shaps_df.to_csv(os.path.join(OUTPUT_FOLDER, 'shaps_df_trained_on_train_stopped_on_val.csv'), index=False)

print('################## FEATURE IMPORTANCE END ##################')
print("################## MODEL REFIT START ##################")

if DO_REFIT:

    train_val_pool = Pool(data=pd.concat([X_train, X_val], axis=0),
                          label=pd.concat([y_train, y_val], axis=0),
                          group_id=pd.concat([X_train, X_val], axis=0)[GROUP_COL],
                          cat_features=CAT_FEATURES,
                          )

    model = get_default_model(tuning=False)
    if TUNE_MODEL:
        print("Using best params from tuned")
        print(best_params)
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

print('################## MODEL REFIT END ##################')
print('################## PREDICTION START ##################')

if MAKE_PREDS:
    from utils import predict_in_format

    subm_df = pd.read_feather(os.path.join(DATA_PATH, 'submission_df_preprocessed.feather'), columns=cols_to_use)
    subm_df.sort_values([GROUP_COL], inplace=True)
    subm_name = 'submission_22'
    subm_filename = f'submissions/{subm_name}.csv'
    subm_scores_filename = f'submissions/{subm_name}_scores.csv'

    prepare_cats(subm_df, CAT_FEATURES)

    subm_pool = Pool(
        data=subm_df,
        group_id=subm_df[GROUP_COL],
        cat_features=CAT_FEATURES,
    )

    output_df = predict_in_format(model, subm_df, subm_pool, GROUP_COL, PREDICT_ITEM_COL)

    os.makedirs(os.path.join(OUTPUT_FOLDER, 'submissions'), exist_ok=True)
    output_df.to_csv(os.path.join(OUTPUT_FOLDER, subm_scores_filename), index=False)
    output_df[[GROUP_COL, 'prop_id']].to_csv(os.path.join(OUTPUT_FOLDER, subm_filename), index=False)

print('################## PREDICTION END ##################')
