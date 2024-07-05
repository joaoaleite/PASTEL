import argparse
import os
import random

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Define the arguments
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--dataset", type=str, required=True)

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    random.seed(42)
    args = parse_arguments()
    DATASET = args.dataset
    MODEL_NAME = args.model_name

    # %%
    CACHE_FOLDER = f"data/cache/"
    CACHE_PATH = os.path.join(CACHE_FOLDER, f"{DATASET}.jsonl")
    DATASET_PATH = f"data/datasets/{DATASET}.csv"
    PROCESSED_FOLDER = "data/signals/"

    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    df_fn = pd.read_csv(DATASET_PATH)
    df_fn = df_fn[["article_id", "title", "text"]]
    df_fn = df_fn.fillna("")
    df_fn["text"] = df_fn.apply(lambda x: x["title"] + "\n" + x["text"], axis=1)
    df_fn = df_fn[["article_id", "text"]]
    total_examples = len(df_fn)

    df = pd.read_json(CACHE_PATH, lines=True)
    df = df.merge(df_fn, on="article_id")
    df = df.drop_duplicates(["article_id"])
    num_processed_examples = len(df)
    num_randomized = len(df[df["objective_pred"] == -1])
    df.loc[df["objective_pred"] == -1, "objective_pred"] = df.loc[df["objective_pred"] == -1]["objective_pred"].apply(
        lambda x: random.randint(0, 1)
    )

    print(f"RANDOMIZED {num_randomized} PREDICTIONS WITH INVALID INFERENCE ({num_randomized/total_examples*100:.1f}%).")
    print(
        f"MISSING {total_examples-num_processed_examples} EXAMPLES FROM THE DATASET "
        + f"{(total_examples-num_processed_examples)/total_examples*100}%."
    )
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(PROCESSED_FOLDER, f"{DATASET}.csv"), index=False)
