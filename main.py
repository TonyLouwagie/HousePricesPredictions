import data_prep

fp = "data/train.csv"
df = data_prep.load_and_clean(fp)
df, _ = data_prep.clean_after_eda(df)
print(df)