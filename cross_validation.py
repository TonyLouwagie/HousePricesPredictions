import numpy as np
from sklearn.model_selection import cross_val_score


def cross_val_aggregate(model, X, y, folds):
    
    scores = cross_val_score(model, X, y, cv=folds)
    
    mean_score = scores.mean()
    sd_score = np.std(scores)
    
    aggregate_scores = {
        "Mean": mean_score,
        "Standard Deviation": sd_score
    }

    return aggregate_scores
    