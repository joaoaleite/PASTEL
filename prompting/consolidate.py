import pandas as pd
import os

DEVICE_NUM=0
MODEL_NAME="orca70b"
DATASET="politifact"
VERBOSE=False

# %%
CACHE_FOLDER = f"data/caches/{DATASET}/{MODEL_NAME}/"
CACHE_PATH = CACHE_FOLDER + "cache.jsonl"
PROCESSED_FOLDER = f"data/processed/{DATASET}/{MODEL_NAME}/"
DATASET_PATH = f"data/datasets/{DATASET}.csv"

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

df_fn = pd.read_csv(DATASET_PATH)
df_fn = df_fn[["article_md5", "title", "text"]]
df_fn = df_fn.fillna("")
df_fn["text"] = df_fn.apply(lambda x: x["title"]+"\n"+x["text"], axis=1)
df_fn = df_fn[["article_md5", "text"]]

df = pd.read_json(CACHE_PATH, lines=True)
df = df.merge(df_fn, on="article_md5")
df = df.drop_duplicates(["article_md5"])
df = df.reset_index(drop=True)
df.to_csv(PROCESSED_FOLDER+f"{DATASET}.csv", index=False)