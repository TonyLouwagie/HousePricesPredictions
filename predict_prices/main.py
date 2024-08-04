import lightgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, BayesianRidge
import xgboost

import cross_validation
import data_prep

train_fp = "predict_prices/data/train.csv"
test_fp = "predict_prices/data/test.csv"
# target variable
tgt = 'SalePrice'
# temporary variable to remove categoricals
include_categoricals = False
# number of folds to cross-validate across
folds = 2
# model to cross validate
lr = LinearRegression()
ohe = False
n_iter = 2

rf = RandomForestRegressor()
rf_param_grid = {
    "n_estimators": tuple(range(10, 1000)),
    "min_samples_split": tuple(range(2, 10)),
    "min_samples_leaf": tuple(range(1, 10))
}

xgb = xgboost.XGBRegressor()
xgb_param_grid = {
    "booster": ['gbtree', 'gblinear']
}

lgbm = lightgbm.LGBMRegressor(verbose=-1)
lgbm_param_grid = {
    'max_depth': tuple(range(1, 100)),
    'num_leaves': tuple(range(1, 100000, 100))
}

en = ElasticNet()
lasso = Lasso()
ridge = Ridge()
br = BayesianRidge()

train_df = data_prep.load_data(train_fp)
train_df = data_prep.eda_clean(train_df)
train_df, _ = data_prep.clean_after_eda(train_df)
ord_enc = data_prep.ordinal_encode(train_df)
train_df = data_prep.ordinal_transform(train_df, ord_enc)
cat_enc = data_prep.categorical_encoder(train_df, ohe)
train_df = data_prep.categorical_transform(train_df, cat_enc)
train_X, train_y = data_prep.split_x_y(train_df, tgt)
train_X = train_X if include_categoricals else data_prep.drop_categoricals(train_X)

lr.fit(train_X, train_y)
lr_aggregate_scores = cross_validation.cross_val_aggregate(lr, train_X, train_y, folds)
lr_param_scores = {
    "model": lr,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": lr_aggregate_scores["Mean"],
    "standard_dev": lr_aggregate_scores["Standard Deviation"]
}

en.fit(train_X, train_y)
en_aggregate_scores = cross_validation.cross_val_aggregate(en, train_X, train_y, folds)
en_param_scores = {
    "model": en,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": en_aggregate_scores["Mean"],
    "standard_dev": en_aggregate_scores["Standard Deviation"]
}

lasso.fit(train_X, train_y)
lasso_aggregate_scores = cross_validation.cross_val_aggregate(lasso, train_X, train_y, folds)
lasso_param_scores = {
    "model": lasso,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": lasso_aggregate_scores["Mean"],
    "standard_dev": lasso_aggregate_scores["Standard Deviation"]
}

ridge.fit(train_X, train_y)
ridge_aggregate_scores = cross_validation.cross_val_aggregate(ridge, train_X, train_y, folds)
ridge_param_scores = {
    "model": ridge,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": ridge_aggregate_scores["Mean"],
    "standard_dev": ridge_aggregate_scores["Standard Deviation"]
}

br.fit(train_X, train_y)
br_aggregate_scores = cross_validation.cross_val_aggregate(br, train_X, train_y, folds)
br_param_scores = {
    "model": br,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": br_aggregate_scores["Mean"],
    "standard_dev": br_aggregate_scores["Standard Deviation"]
}

rf = cross_validation.bayes_cross_validation(rf, train_X, train_y, rf_param_grid, n_iter)
rf_aggregate_scores = cross_validation.cross_val_aggregate(rf, train_X, train_y, folds)
rf_param_scores = {
    "model": rf,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": rf_aggregate_scores["Mean"],
    "standard_dev": rf_aggregate_scores["Standard Deviation"]
}

xgb = cross_validation.bayes_cross_validation(xgb, train_X, train_y, xgb_param_grid, n_iter)
xgb_aggregate_scores = cross_validation.cross_val_aggregate(xgb, train_X, train_y, folds)
xgb_param_scores = {
    "model": xgb,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": xgb_aggregate_scores["Mean"],
    "standard_dev": xgb_aggregate_scores["Standard Deviation"]
}

lgbm = cross_validation.bayes_cross_validation(lgbm, train_X, train_y, lgbm_param_grid, n_iter)
lgbm_aggregate_scores = cross_validation.cross_val_aggregate(lgbm, train_X, train_y, folds)
lgbm_param_scores = {
    "model": lgbm,
    "ordinal_encoder": ord_enc,
    "categorical_encoder": cat_enc,
    "score": lgbm_aggregate_scores["Mean"],
    "standard_dev": lgbm_aggregate_scores["Standard Deviation"]
}

models = [lr_param_scores, rf_param_scores, xgb_param_scores, lgbm_param_scores, en_param_scores, lasso_param_scores]

param_scores = pd.DataFrame(models).sort_values('score',ascending=False)

print(param_scores[['model','score','standard_dev']])

# take only first row in case of ties (I don't care which model if they're tied)
champ = param_scores[param_scores.score == param_scores.score.max()].reset_index()
champ_model = champ.model[0]
champ_ord_enc = champ.ordinal_encoder[0]
champ_cat_enc = champ.categorical_encoder[0]

test_df = data_prep.load_data(test_fp)
test_df = data_prep.eda_clean(test_df)
test_df, _ = data_prep.clean_after_eda(test_df)
test_df = data_prep.ordinal_transform(test_df, champ_ord_enc)
test_df = data_prep.categorical_transform(test_df, champ_cat_enc)
test_X = test_df if include_categoricals else data_prep.drop_categoricals(test_df)

# the model was trained against log transformed target. Invert log for predictions
test_y = np.exp(champ_model.predict(test_X))

print(test_y[:4])
