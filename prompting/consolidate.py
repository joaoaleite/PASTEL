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

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

df = pd.read_json(CACHE_PATH, lines=True)
df = df.drop_duplicates(["article_md5"])
df = df.reset_index(drop=True)
df.to_csv(PROCESSED_FOLDER+f"{DATASET}.csv")