from sklearn.linear_model import LinearRegression

import cross_validation
import data_prep

fp = "predict_prices/data/train.csv"
# target variable
tgt = 'SalePrice'
# temporary variable to remove categoricals
# TODO: get rid of this once we have handling to remove all categoricals
include_categoricals = False
# number of folds to cross-validate across
folds = 4
# model to cross validate
lr = LinearRegression()

df = data_prep.load_data(fp)
df = data_prep.eda_clean(df)
df, _ = data_prep.clean_after_eda(df)
df, _ = data_prep.ordinal_encode(df)
X, y = data_prep.split_x_y(df, tgt, include_categoricals)

aggregate_scores = cross_validation.cross_val_aggregate(lr, X, y, folds)

print(aggregate_scores)
