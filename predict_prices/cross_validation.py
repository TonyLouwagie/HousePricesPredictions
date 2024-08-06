import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score # type: ignore
from skopt import BayesSearchCV # type: ignore


def cross_val_aggregate(model, X: pd.DataFrame, y: pd.Series, folds: int) -> dict[str, float]:
    """
    Run cross validation and aggregate scores on one model
    :param model: the model to score
    :param X: explanatory variables
    :param y: target variable
    :param folds: the number of folds to cross validate over
    :return: aggregate_scores: aggregated cross validation scores
    """

    scores = cross_val_score(model, X, y, cv=folds)

    mean_score = scores.mean()
    sd_score = np.std(scores)

    aggregate_scores = {
        "Mean": mean_score,
        "Standard Deviation": sd_score
    }

    return aggregate_scores


def save_model_performance_parameters(model, train_X, train_y, folds, ord_enc, cat_enc):
    aggregate_scores = cross_val_aggregate(model, train_X, train_y, folds)
    param_scores = {
        "model": model,
        "ordinal_encoder": ord_enc,
        "categorical_encoder": cat_enc,
        "score": aggregate_scores["Mean"],
        "standard_dev": aggregate_scores["Standard Deviation"]
    }

    return param_scores


def bayes_cross_validation(model, train_X: pd.DataFrame, train_y: pd.Series, param_grid: dict, n_iter: int):
    """
    Cross validation with a hyperparameter grid so we can tune to the best hyperparameters for each model
    :param model:
    :param train_X:
    :param train_y:
    :param param_grid:
    :param n_iter:
    :return: best
    """
    best = BayesSearchCV(estimator=model,
                         search_spaces=param_grid,
                         cv=4,
                         n_jobs=4,
                         n_iter=n_iter,
                         random_state=42)
    best.fit(train_X, train_y)

    return best
