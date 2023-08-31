import pandas as pd
import os

DEVICE_NUM=0
MODEL_SIZE=7
DATASET="fakenewsnet"
VERBOSE=False

# %%
CACHE_FOLDER = f"data/caches/{MODEL_SIZE}/"
CACHE_PATH = CACHE_FOLDER + "cache.jsonl"
PROCESSED_FOLDER = "data/processed/"

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

df = pd.read_json(CACHE_PATH, lines=True)
df = df.drop_duplicates(["article_md5"])
df = df.reset_index(drop=True)
df.to_csv(PROCESSED_FOLDER+DATASET+"_processed.csv")