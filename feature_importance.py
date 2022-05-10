import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import shap
from catboost import Pool


def metric_with_perm_column(model, X, y, perm_col, metric, **kwargs):
    save = X[perm_col].copy()
    X[perm_col] = np.random.permutation(X[perm_col])
    m = metric(model, X, y, **kwargs)
    X[perm_col] = save
    return m


def permutation_importances(model, X, y, metric, **kwargs):
    """
    Calculates feature importance for the model which was trained to predict the difference between test sales mean and
    actual value

    Args:
        model:
        X:
        y:
        #test_sales_mean:
        metric:

    Returns:

    """
    baseline = metric(model, X, y, **kwargs)

    imp = []
    for col in tqdm(X.columns, total=len(X.columns)):
        m = metric_with_perm_column(model, X, y, col, metric, **kwargs)
        imp.append(m - baseline)

    #     import multiprocessing
    #     from multiprocessing import Pool

    #     arg_tuples = [(model, X, y, test_sales_mean, col, metric) for col in X.columns]
    #     print('arg tuples prepared')

    #     try:
    #         pool = Pool(multiprocessing.cpu_count()-1)
    #         imp = []
    #         for result in tqdm(pool.map(metric_with_perm_column, arg_tuples), total=len(arg_tuples)):
    #             imp.append(result)
    #         pool.terminate()
    #         pool.join()
    #     except KeyboardInterrupt:
    #         pool.terminate()
    #         pool.join()

    return np.array(imp)


def get_and_plot_feature_imp_catboost(model, X, y, method, top_n=None, **kwargs):
    """

    Args:
        model:
        X:
        y:
        method (str): may be one of ['Permutation', 'ShapValues', 'LossFunctionChange', 'PredictionValuesChange', 'SHAP']. 'LossFunctionChange': difference between the metric (Loss function) obtained using the model in normal scenario (when we include the feature) and model without this feature (model is built approximately using the original model with this feature removed from all the trees in the ensemble). Higher the difference, the more important the feature is. 'PredictionValuesChange': shows how much on average the prediction changes if the feature value changes. The bigger the value of the importance the bigger on average is the change to the prediction value, if this feature is changed. Normalized so that the sum of importances of all features (all are positive) is equal to 100.
        top_n:
        **kwargs:

    Returns:

    """
    if method == "Permutation":
        fi = permutation_importances(model=model, X=X, y=y, **kwargs)

    elif method == "SHAP":
        shap_values_raw = model.get_feature_importance(Pool(X, label=y, **kwargs),
                                                       type="ShapValues")
        shap_values = shap_values_raw[:, :-1]
        shap.summary_plot(shap_values, X, max_display=top_n)

        feature_score_raw = pd.DataFrame(list(zip(X.columns, abs(shap_values).mean(axis=0))),
                                         columns=['Feature', 'Score'])
        # shap_values_raw may be useful further
        return feature_score_raw, shap_values_raw

    else:
        fi = model.get_feature_importance(Pool(X, label=y, **kwargs),
                                          type=method)
    if method != "SHAP":
        feature_score_raw = pd.DataFrame(list(zip(X.columns, fi)),
                                         columns=['Feature', 'Score'])

        feature_score = feature_score_raw.sort_values(
            by='Score', ascending=False)

        if top_n is not None:
            feature_score = feature_score.head(top_n)
            # 26/11/2020 Oleg removed zeros
            feature_score = feature_score[feature_score['Score'] != 0]

        plt.rcParams["figure.figsize"] = (7, len(feature_score) / 4)
        ax = feature_score.plot('Feature', 'Score', kind='barh', color='c')
        ax.set_title("Feature Importance using {}".format(method), fontsize=14)
        ax.set_xlabel("features")
        ax.invert_yaxis()
        plt.show()

        return feature_score_raw
