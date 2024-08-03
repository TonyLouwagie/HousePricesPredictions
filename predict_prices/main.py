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

train_df = data_prep.load_data(train_fp)
train_df = data_prep.eda_clean(train_df)
train_df, _ = data_prep.clean_after_eda(train_df)
# TODO: Save enc into the underscore when we need to use it to transform test data
ord_enc = data_prep.ordinal_encode(train_df)
train_df = data_prep.ordinal_transform(train_df, ord_enc)
train_df, _ = data_prep.categorical_encoder(train_df, ohe)
train_X, train_y = data_prep.split_x_y(train_df, tgt, include_categoricals)

aggregate_scores = cross_validation.cross_val_aggregate(lr, train_X, train_y, folds)
print(aggregate_scores)

test_df = data_prep.load_data(test_fp)
test_df = data_prep.eda_clean(test_df)
test_df, _ = data_prep.clean_after_eda(test_df)
test_df = data_prep.ordinal_transform(test_df, ord_enc)

print(test_df.head())
