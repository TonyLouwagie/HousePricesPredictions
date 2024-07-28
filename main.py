import data_prep

fp = "data/train.csv"
df = data_prep.load_and_clean(fp)
print(df)