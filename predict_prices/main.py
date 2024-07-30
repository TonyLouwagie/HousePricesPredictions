from sklearn.linear_model import LinearRegression

import cross_validation
import data_prep

fp = "predict_prices/data/train.csv"
tgt = 'SalePrice'
include_categoricals = False
folds = 4
lr = LinearRegression()

df = data_prep.load_and_clean(fp)
df, _ = data_prep.clean_after_eda(df)
X, y = data_prep.split_x_y(df, tgt, include_categoricals)

aggregate_scores = cross_validation.cross_val_aggregate(lr, X, y, folds)

print(aggregate_scores)
