import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import cross_validation
import data_prep

train_fp = "predict_prices/data/train.csv"
test_fp = "predict_prices/data/test.csv"
# target variable
tgt = 'SalePrice'
# temporary variable to remove categoricals
include_categoricals = False
# number of folds to cross-validate across
folds = 4
# model to cross validate
lr = LinearRegression()
ohe = False

rf = RandomForestRegressor()
rf_param_grid = {
    "n_estimators": tuple(range(10, 1000)),
    "min_samples_split": tuple(range(2, 10)),
    "min_samples_leaf": tuple(range(1, 10))
}
n_iter = 20

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
rf = cross_validation.bayes_cross_validation(rf, train_X, train_y, rf_param_grid, n_iter)

lr_aggregate_scores = cross_validation.cross_val_aggregate(lr, train_X, train_y, folds)
rf_aggregate_scores = cross_validation.cross_val_aggregate(rf, train_X, train_y, folds)

print("linear regression:", lr_aggregate_scores)
print("random forest:", rf_aggregate_scores)

test_df = data_prep.load_data(test_fp)
test_df = data_prep.eda_clean(test_df)
test_df, _ = data_prep.clean_after_eda(test_df)
test_df = data_prep.ordinal_transform(test_df, ord_enc)
test_df = data_prep.categorical_transform(test_df, cat_enc)
test_X = test_df if include_categoricals else data_prep.drop_categoricals(test_df)

# the model was trained against log transformed target. Invert log for predictions
test_y = np.exp(lr.predict(test_X))

print(test_y[:4])
