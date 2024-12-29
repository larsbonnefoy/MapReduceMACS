import pandas as pd

df = pd.read_csv("Iris.csv")

# remove id column

# convert all columns to floats
df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] = df[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
].astype(float)

# select training values (where species exist)
df_train = df[df["Species"].notna()]


# store classes and Ids for later
df_train_classes = df_train["Species"]
train_ids = df_train["Id"]

# remove classes to compute min-max
df_train = df_train.drop(columns=["Species", "Id"])

# get min and max
train_min = df_train.min()
train_max = df_train.max()

df_train_scaled = (df_train - train_min) / (train_max - train_min)

# add classes once scaling has been done
combined = pd.concat([train_ids, df_train_scaled, df_train_classes], axis=1)
combined.to_csv("Iris_train.csv", index=False)

df_test = df[df["Species"].isna()]
# save ID for later, need to be removed to compute min/max
test_ids = df_test["Id"]
df_test = df_test.drop(columns=["Species", "Id"])
df_test_scaled = (df_test - train_min) / (train_max - train_min)

test_combined = pd.concat([test_ids, df_test_scaled], axis=1)
test_combined.to_csv("Iris_test.csv", index=False)
