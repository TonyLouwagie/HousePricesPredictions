import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


def cross_val_aggregate(model, X: pd.DataFrame, y: pd.Series, folds: int):
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
