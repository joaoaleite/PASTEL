# %%
import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class llama2_platypus:
    def __init__(self, size, model_name=None):
        if model_name is None:
            if size in [7, 13, 70]:
                model_name = f"garage-bAInd/Platypus2-{size}B"
            else:
                raise Exception(f"Size {size} not available for Llama. Choose 7, 13 or 70.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, add_bos_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model = self.model.eval()
        self.device = self.model.device

    def prompt(self, input, question, system_context, max_new_tokens):
        # The number of tokens for the question and prompt formatting amounts to 33 tokens
        # so we can use 4064 tokens for the input text. Will use 4050 to leave some room.
        truncate_to = 4050 - max_new_tokens
        input = self.tokenizer.decode(
            self.tokenizer.encode(
                input,
                return_tensors="pt",
                truncation=True,
                max_length=truncate_to,
                add_special_tokens=False,
            )[0]
        )  # ensure the text will fit 4096 tokens
        prompt = (
            f"### Instruction:\n{system_context}\n\n### Input:\n{input.strip()}\n\n{question.strip()}\n\n###"
            " Response:\n"
        )
        # ans1 = self.get_next_word_probs(prompt, allow_abstain)
        ans = self.model.generate(
            self.tokenizer.encode(prompt, return_tensors="pt").to(self.device),
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        ans = self.tokenizer.decode(ans[0])
        ans = ans.split("### Response:")[1].strip()
        return ans


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
    abstain_context = (
        "You are expeted to answer with 'Yes' or 'No', but you are also allowed to answer with 'Unsure' if you do not"
        " have enough information or context to provide a reliable answer."
    )

    system_context = (
        "You are a helpful and unbiased news verification assistant. You will be provided with the"
        " title and the full body of text of a news article. Then, you will answer further questions related"
        " to the given article. Ensure that your answers are grounded in reality,"
        f" truthful and reliable.{abstain_context}"
    )

    prompt = """{title}\n{text}"""
    preds = []
    trues = []
    num_abstain = 0
    num_no = 0
    num_yes = 0
    with tqdm(total=len(df) * len(signal_df)) as pbar:
        for i, article_row in enumerate(df.itertuples()):
            if article_row.article_id in [row["article_id"] for row in load_cache(CACHE_PATH)]:
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
                if rationales:
                    processed[question_row._2 + "_rationale"] = answer_ws
                pbar.update(1)

            processed["objective_pred"] = category_zs
            processed["objective_true"] = true
            if rationales:
                processed["rationale_zs"] = answer_zs if rationales else ""
            processed["article_id"] = article_row.article_id

            dump_cache(processed, CACHE_PATH)
            pbar.update(1)


# %%
if __name__ == "__main__":
    args = parse_arguments()
    VERBOSE = args.verbose
    MODEL_SIZE = args.model_size
    MODEL_NAME = args.model_name
    DEVICE_NUM = args.device_num
    RATIONALES = args.rationales
    DATASET = args.dataset
    CACHE_FOLDER = "data/cache"
    CACHE_PATH = os.path.join(CACHE_FOLDER, f"{DATASET}.jsonl")
    DATASET_PATH = f"data/datasets/{DATASET}.csv"
    SIGNALS_PATH = "data/signals.csv"

    assert os.path.exists(DATASET_PATH)
    assert os.path.exists(SIGNALS_PATH)
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)

    df = pd.read_csv(DATASET_PATH)
    # df = df.sample(frac=1)  # randomize for more effective parallel processing
    signal_df = pd.read_csv(SIGNALS_PATH)

    # %%
    if MODEL_NAME == "llama2_platypus":
        model = llama2_platypus(size=MODEL_SIZE)
    else:
        raise Exception(f"No model named {MODEL_NAME} with size {MODEL_SIZE}")

    print(f"Dataset: {DATASET} Model Name: {MODEL_NAME} Model Size: {MODEL_SIZE}")
    print("Device Name:", torch.cuda.get_device_name())
    process(model, df, signal_df, verbose=VERBOSE, rationales=RATIONALES)
