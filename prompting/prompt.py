# %%
import os
import pandas as pd
from utils import llama2_platypus
import torch
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import argparse
import random


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Define the arguments
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--model_size", type=int, choices=[7, 13, 70], required=True)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--rationales", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Parse the arguments
    args = parser.parse_args()

    return args


def category_mapping(answer):
    if answer.lower().startswith("no") or answer.lower().startswith("false"):
        category = 0
    elif answer.lower().startswith("yes") or answer.lower().startswith("true"):
        category = 1
    else:
        category = -1

    return category


def load_cache(p):
    cache = []
    if os.path.exists(p):
        with open(p, "r") as f:
            for i, line in enumerate(f):
                cache.append(json.loads(line))

    return cache


def dump_cache(line, p):
    with open(p, "a") as f:
        f.write(json.dumps(line) + "\n")


# %%
def process(model, df, signal_df, verbose=False, rationales=False):
    system_context = """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Then, you will answer further questions related to the given article. Ensure that your answers are grounded in reality, truthful and reliable.{abstain_context}"""

    prompt = """{title}\n{text}"""
    abstain_context = (
        " You are expeted to answer with 'Yes' or 'No', but you are also allowed to answer with 'Unsure' if you do not"
        " have enough information or context to provide a reliable answer."
    )
    preds = []
    trues = []
    num_abstain = 0
    num_no = 0
    num_yes = 0
    with tqdm(total=len(df) * len(signal_df)) as pbar:
        for i, article_row in enumerate(df.itertuples()):
            if article_row.article_md5 in [row["article_md5"] for row in load_cache(CACHE_PATH)]:
                continue

            # ZS Question
            system_context_zs = system_context.format(abstain_context="")
            system_context_zs += " Afterwards, explain your answer by providing a rationale." if rationales else ""
            input_zs = prompt.format(title=article_row.title, text=article_row.text)
            question_zs = "Does this article contain misinformation? (Yes/No)"
            try:
                answer_zs = model.prompt(
                    input=input_zs,
                    question=question_zs,
                    system_context=system_context_zs,
                    max_new_tokens=256 if rationales else 1,
                )
            except Exception as e:
                print("ERROR", e)
                continue

            category_zs = category_mapping(answer_zs)
            preds.append(category_zs)
            true = article_row.objective
            trues.append(true)

            num_yes += 1 if category_zs == 1 else 0
            num_no += 1 if category_zs == 0 else 0
            num_abstain += 1 if category_zs == -1 else 0

            acc = accuracy_score(trues, preds)
            f1 = f1_score(trues, preds, average="macro")
            updated_description = (
                f"Acc={acc*100:.2f}, F1={f1*100:.2f}, Total={i}, Num_Yes={num_yes}, Num_No={num_no},"
                f" Num_Abstain={num_abstain}"
            )
            pbar.set_description(updated_description)

            if verbose:
                print(answer_zs)

                print(10 * "-")
                print("Objective:", article_row.objective)

            processed = {}
            for j, question_row in enumerate(signal_df.itertuples()):
                system_context_ws = system_context.format(abstain_context=abstain_context)
                system_context_ws += " Afterwards, explain your answer by providing a rationale." if rationales else ""
                input_ws = prompt.format(title=article_row.title, text=article_row.text)
                question_ws = question_row.Question + " (Yes/Unsure/No)"
                try:
                    answer_ws = model.prompt(
                        input=input_ws,
                        question=question_ws,
                        system_context=system_context_ws,
                        max_new_tokens=256 if rationales else 1,
                    )
                except Exception as e:
                    print("ERROR", e)
                    break

                category_ws = category_mapping(answer_ws)
                if verbose:
                    print(question_row.Question, category_ws)
                    print(answer_ws)

                processed[question_row._2] = category_ws
                processed[question_row._2 + "_rationale"] = answer_ws
                pbar.update(1)

            processed["objective_pred"] = category_zs
            processed["objective_true"] = true
            processed["rationale_zs"] = answer_zs if rationales else ""
            processed["article_md5"] = article_row.article_md5

            if len(processed.keys()) == 22:
                dump_cache(processed, CACHE_PATH)
            pbar.update(1)


# %%
if __name__ == "__main__":
    args = parse_arguments()
    DATASET = args.dataset
    VERBOSE = args.verbose
    MODEL_SIZE = args.model_size
    MODEL_NAME = args.model_name
    DEVICE_NUM = args.device_num
    RATIONALES = args.rationales

    CACHE_FOLDER = f"data/caches/{DATASET}/{MODEL_NAME}/{MODEL_SIZE}"
    CACHE_PATH = os.path.join(CACHE_FOLDER, "cache_rationales.jsonl")
    DATASET_PATH = f"data/datasets/{DATASET}.csv"
    SIGNALS_PATH = "data/signals.csv"

    assert os.path.exists(DATASET_PATH)
    assert os.path.exists(SIGNALS_PATH)
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)

    df = pd.read_csv(DATASET_PATH)
    df = df.sample(frac=1)  # randomize for more effective parallel processing
    signal_df = pd.read_csv(SIGNALS_PATH)

    # %%
    if MODEL_NAME == "llama2_platypus":
        model = llama2_platypus(size=MODEL_SIZE)
    else:
        raise Exception(f"No model named {MODEL_NAME} with size {MODEL_SIZE}")

    print(f"Dataset: {DATASET} Model Name: {MODEL_NAME} Model Size: {MODEL_SIZE}")
    print("Device Name:", torch.cuda.get_device_name())
    process(model, df, signal_df, verbose=VERBOSE, rationales=RATIONALES)
