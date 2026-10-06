"""Quick first look at the data."""
import pandas as pd

DATA = "playground-series-s6e10"
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

print("train", train.shape, "test", test.shape)
print("\nTarget balance:\n", train["satisfaction"].value_counts(normalize=True))
print("\nMissing values (train):\n", train.isna().sum()[lambda s: s > 0])
print("\nMissing values (test):\n", test.isna().sum()[lambda s: s > 0])
print("\nDtypes / unique counts:")
print(pd.DataFrame({"dtype": train.dtypes, "nunique": train.nunique()}))
print("\nNumeric summary:\n", train.describe().T[["mean", "std", "min", "max"]])
print("\nCategorical levels:")
for c in ["Gender", "Customer Type", "Type of Travel", "Class"]:
    print(c, train[c].value_counts().to_dict())
print("\nDuplicate feature rows in train:", train.drop(columns=["id", "satisfaction"]).duplicated().sum())
