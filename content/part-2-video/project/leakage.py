import pandas as pd

train_df = pd.read_csv("data/ufc10/metadata/train.csv")
val_df   = pd.read_csv("data/ufc10/metadata/val.csv")

# Ensure columns are what we expect
print(train_df.head())
print(val_df.head())
    
overlap = set(train_df["video_name"]) & set(val_df["video_name"])
print(f"Number of overlapping videos: {len(overlap)}")

if len(overlap) > 0:
    print("Example overlaps:", list(overlap)[:10])


print(train_df["label"].value_counts(normalize=True).sort_index())
print(val_df["label"].value_counts(normalize=True).sort_index())
