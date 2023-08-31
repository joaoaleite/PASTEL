# %%
# Parameters
DEVICE_NUM=4
MODEL_SIZE=70
DATASET="fakenewsnet"
VERBOSE=False

# %%


# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_NUM)
CACHE_PATH = f"../data/caches/{MODEL_SIZE}/cache.jsonl"
DATASET_PATH = f"../data/{DATASET}.csv"
SIGNALS_PATH = "../data/signals.csv"

assert os.path.exists(DATASET_PATH)
assert os.path.exists(SIGNALS_PATH)
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# %%
import pandas as pd
from utils import llama_chat_hf
import torch
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import hashlib

# %%
print("Device Name:", torch.cuda.get_device_name(), "Device Number:", DEVICE_NUM)

# %%
df = pd.read_csv(DATASET_PATH)
signal_df = pd.read_csv(SIGNALS_PATH)

# %%
model = llama_chat_hf(size=MODEL_SIZE)

# %%
system_context = \
    """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Then, you will answer further questions related to the given article. Ensure that your answers are grounded in reality, truthful and reliable. {abstain_context}It is essential that you only answer objectively with one of the following options: {options}. Please do not answer with anything other than the option provided."""

prompt = """{title}\n{text}\n\n{question} ({options})"""
# abstain_context = "You are only allowed to answer with 'Unsure' if you do not have enough information or context to provide a reliable answer."
abstain_context = ""
# fake_news_context = "Fake news is false or inaccurate information, especially that which is deliberately intended to deceive." # Decreases performance.

# %%
def category_mapping(answer):
    if answer.lower().startswith("no"):
        category = 0
    elif answer.lower().startswith("yes"):
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
        f.write(json.dumps(line)+"\n")

# %%
def process(verbose=False):
    preds = []
    trues = []
    processed_records = []
    with tqdm(total=len(df)*len(signal_df)) as pbar:
        for i, article_row in enumerate(df.itertuples()):
            if article_row.article_md5 in [row["article_md5"] for row in load_cache(CACHE_PATH)]:
                continue

            # ZS Question
            system_context_zs = system_context.format(options="Yes/No", abstain_context="")
            prompt_formatted = prompt.format(title=article_row.title, text=article_row.text, system_context=system_context_zs, question="Does this article contain misinformation?", options="Yes/No")
            try:
                answer_zs = model.prompt(prompt_formatted, allow_abstain=False)
            except torch.cuda.OutOfMemoryError as e:
                continue # stop processing this example
            
            category_zs = category_mapping(answer_zs)
            preds.append(category_zs)
            true = article_row.objective
            trues.append(true)

            acc = accuracy_score(trues, preds)
            f1 = f1_score(trues, preds, average="macro")
            num_yes = len([x for x in preds if x == 1])
            num_no = len([x for x in preds if x == 0])
            num_abstain = len([x for x in preds if x == -1])
            updated_description = f"Acc={acc*100:.2f}, F1={f1*100:.2f}, Total={i}, Num_Yes={num_yes}, Num_No={num_no}, Num_Abstain={num_abstain}"
            pbar.set_description(updated_description)
            
            if verbose:
                print(answer_zs)
                
                print(10*"-")
                print("Objective:", article_row.objective)

            processed = {}
            for j, question_row in enumerate(signal_df.itertuples()):
                system_context_ws = system_context.format(options="Yes/Unsure/No", abstain_context=abstain_context)
                prompt_formatted_ws = prompt.format(title=article_row.title, text=article_row.text, question=question_row.Question, options="Yes/Unsure/No")

                try:
                    answer_ws = model.prompt(prompt_formatted_ws, system_context=system_context_ws, allow_abstain=True)
                except torch.cuda.OutOfMemoryError as e:
                    break # stop processing this example and the next questions

                if verbose:
                    print(question_row.Question, category_ws)
                    print(answer_ws)
                    
                category_ws = category_mapping(answer_ws)
                processed[question_row._2] = category_ws
                pbar.update(1)
            
            processed["objective_pred"] = category_zs
            processed["objective_true"] = true
            processed["article_md5"] = article_row.article_md5
            dump_cache(processed, CACHE_PATH)
            pbar.update(1)

# %%
process(verbose=VERBOSE)

# %%
preds = []
trues = []
processed_records = []
with tqdm(total=len(df)*len(signal_df)) as pbar:
    for i, article_row in enumerate(df.itertuples()):
        if article_row.article_md5 in [row["article_md5"] for row in load_cache(CACHE_PATH)]:
            continue

        # ZS Question
        system_context_zs = system_context.format(options="Yes/No", abstain_context="")
        prompt_formatted = prompt.format(title=article_row.title, text=article_row.text, system_context=system_context_zs, question="Does this article contain misinformation?", options="Yes/No")
        try:
            answer_zs = model.prompt(prompt_formatted, allow_abstain=False)
        except torch.cuda.OutOfMemoryError as e:
            continue # stop processing this example
        
        category_zs = category_mapping(answer_zs)
        preds.append(category_zs)
        label_converter = lambda x: 0 if x == "real" else 1
        true = label_converter(article_row.objective)
        trues.append(true)

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")
        num_yes = len([x for x in preds if x == 1])
        num_no = len([x for x in preds if x == 0])
        num_abstain = len([x for x in preds if x == -1])
        updated_description = f"Acc={acc*100:.2f}, F1={f1*100:.2f}, Total={i}, Num_Yes={num_yes}, Num_No={num_no}, Num_Abstain={num_abstain}"
        pbar.set_description(updated_description)
        
        print(answer_zs)
        
        print(10*"-")
        print("Objective:", article_row.objective)

        processed = {}
        for j, question_row in enumerate(signal_df.itertuples()):
            system_context_ws = system_context.format(options="Yes/Unsure/No", abstain_context=abstain_context)
            prompt_formatted_ws = prompt.format(title=article_row.title, text=article_row.text, question=question_row.Question, options="Yes/Unsure/No")

            try:
                answer_ws = model.prompt(prompt_formatted_ws, system_context=system_context_ws, allow_abstain=True)
            except torch.cuda.OutOfMemoryError as e:
                break # stop processing this example and the next questions

            category_ws = category_mapping(answer_ws)
            print(question_row.Question, category_ws)
            print(answer_ws)
            processed[question_row._2] = category_ws
            pbar.update(1)
        
        processed["objective_pred"] = category_zs
        processed["objective_true"] = true
        processed["article_md5"] = article_row.article_md5
        dump_cache(processed, CACHE_PATH)
        pbar.update(1)


