import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from snorkel.labeling.model import LabelModel
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import set_peft_model_state_dict
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import wandb
import torch
import os
import random
from tqdm import tqdm

SEED = 42
pl.seed_everything(SEED, workers=True)

class llama2_platypus():
    def __init__(self, size, model_name=None):
        if model_name is None:
            if size in [7, 13, 70]:
                model_name = f"garage-bAInd/Platypus2-{size}B"
            else:
                raise Exception(f"Size {size} not available for Llama. Choose 7, 13 or 70.")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, add_bos_token=True)
        self.model  = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            # load_in_8bit=True,
            device_map="auto"
        )
        self.model = self.model.eval()
        self.device = self.model.device

    def prompt(self, input, question, system_context):
        # The number of tokens for the question and prompt formatting amounts to 33 tokens
        # so we can use 4064 tokens for the input text. Will use 4050 to leave some room.
        truncate_to = 4050
        input = self.tokenizer.decode(self.tokenizer.encode(input, return_tensors='pt', truncation=True, max_length=truncate_to, add_special_tokens=False)[0]) # ensure the text will fit 4096 tokens
        prompt = f"### Instruction:\n{system_context}\n\n### Input:\n{input.strip()}\n\n{question.strip()}\n\n### Response:\n"
        # ans1 = self.get_next_word_probs(prompt, allow_abstain)
        ans = self.model.generate(self.tokenizer.encode(prompt, return_tensors='pt').to(self.device), max_new_tokens=1, num_beams=1, do_sample=False)
        ans = self.tokenizer.decode(ans[0])
        ans = ans.split("### Response:")[1].strip()
        return ans
    
def aux_get_train_test_fold(fold, dataset, model_size, model_name="llama2_platypus", num_splits=10):
    assert fold < num_splits
    
    dataset_path = f"data/processed/{dataset}/{model_name}/{model_size}/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=SEED)
    for j, (train_idxs, test_idxs) in enumerate(skf.split(range(len(df)), y=df["objective_true"].to_numpy())):
        train_df, test_df = df.iloc[train_idxs], df.iloc[test_idxs]

        if fold == j:
            return train_df, test_df
        
def get_train_test_fold(fold, train_dataset, test_dataset, model_size, model_name="llama2_platypus"):
    train_df, _ = aux_get_train_test_fold(fold=fold, dataset=train_dataset, model_name=model_name, model_size=model_size)
    _, test_df = aux_get_train_test_fold(fold=fold, dataset=test_dataset, model_name=model_name, model_size=model_size)

    return train_df, test_df

        
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--training_method", type=str, choices=["ws", "ft"])
    args = parser.parse_args()
    
    return args


def main(fold, training_method, train_dataset, test_dataset, model_size, model_name):
    experiment_config = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "model_size": model_size,
        "model_name": model_name,
        "training_method": training_method,
        "fold": fold
    }


    wandb.init(project="prompted_credibility_cross_dataset", name=f"{training_method}-TRAIN:{train_dataset}-TEST:{test_dataset}")
    for k, v in experiment_config.items():
        wandb.config[k] = v

    wandb.define_metric('val/f1_macro', summary='max')
    wandb.define_metric('val/acc', summary='max')
    wandb.define_metric('val/precision', summary='max')
    wandb.define_metric('val/recall', summary='max')
    wandb.define_metric('val/loss', summary='min')
    wandb.define_metric('val/false_positive_rate', summary='min')
    wandb.define_metric('val/true_negative_rate', summary='max')
    wandb.define_metric('val/false_negative_rate', summary='min')
    wandb.define_metric('val/true_positive_rate', summary='max')

    df_train, df_test = get_train_test_fold(fold=fold, train_dataset=train_dataset, test_dataset=test_dataset, model_size=model_size, model_name=model_name)
    
    X_train = df_train["text"].tolist()
    X_test = df_test["text"].tolist()
    y_test_gold = df_test["objective_true"].to_numpy()

    if training_method == "ws":
        # Train label model and infer test labels (label model approach)
        L_ws_train = df_train.iloc[:, :19].to_numpy()
        L_ws_test = df_test.iloc[:, :19].to_numpy()
        label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
        label_model.fit(L_ws_train, n_epochs=500, seed=SEED)
        y_pred_ws = label_model.predict(L=L_ws_test, tie_break_policy="random")
        val_acc = accuracy_score(y_test_gold, y_pred_ws)
        val_f1_macro = f1_score(y_test_gold, y_pred_ws, average='macro')

        tn, fp, fn, tp = confusion_matrix(y_test_gold, y_pred_ws).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        precision = precision_score(y_test_gold, y_pred_ws)
        recall = recall_score(y_test_gold, y_pred_ws)

    elif training_method == "ft":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        model_path = f"finetuning/results-{train_dataset}-{fold}/"
        latest_step = sorted([folder.split("-")[1] for folder in os.listir(model_path)], reverse=True).pop()
        best_model_path = os.path.join(model_path, latest_step, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)

        print("loading checkpoint:", best_model_path)
        set_peft_model_state_dict(model, adapters_weights)  # set LoRA weights

        inference_model = llama2_platypus(size=MODEL_SIZE, model=model)
        system_context = """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Then, you will answer further questions related to the given article. Ensure that your answers are grounded in reality, truthful and reliable."""
        prompt = """{title}\n{text}"""
        randomizer = lambda: random.randint(0, 1)
        def class_mapper(answer):
            if answer.lower().startswith("no") or answer.lower().startswith("false"):
                category = 0
            elif answer.lower().startswith("yes") or answer.lower().startswith("true"):
                category = 1
            else:
                category = -1

            return category


        preds = []
        targets = []
        for i, article_row in tqdm(enumerate(df_test.sample(frac=1).itertuples()), total=len(df_test)):
            system_context_zs = system_context.format(abstain_context="")
            input = prompt.format(title=article_row.title, text=article_row.text)
            question = "Does this article contain misinformation? (Yes/No)"
            ans = inference_model.prompt(input=input, question=question, system_context=system_context_zs)
        
            label = class_mapper(ans)

            preds.append(label)
            targets.append(article_row.objective)

        num_invalid = len([v for v in preds if v == -1])
        preds = [v if v != -1 else randomizer() for v in preds]

        val_acc = accuracy_score(targets, preds)
        val_f1_macro = f1_score(targets, preds, average='macro', zero_division=0.0)

        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        precision = precision_score(targets, preds, zero_division=0.0)
        recall = recall_score(targets, preds, zero_division=0.0)

    else:
        raise Exception("Training method not supported.")
    
    wandb.log({
            "val/acc": val_acc,
            "val/f1_macro": val_f1_macro,
            "val/true_positive_rate": true_positive_rate,
            "val/false_positive_rate": false_positive_rate,
            "val/true_negative_rate": true_negative_rate,
            "val/false_negative_rate": false_negative_rate,
            "val/precision": precision,
            "val/recall": recall
    }, step=0)
    
if __name__ == "__main__":
    args = parse_arguments()
    TRAIN_DATASET = args.train_dataset
    FOLD = args.fold
    TEST_DATASET = args.test_dataset
    MODEL_SIZE = 70
    MODEL_NAME = args.model_name
    TRAINING_METHOD = args.training_method

    main(
        fold=FOLD,
        training_method=TRAINING_METHOD,
        train_dataset=TRAIN_DATASET,
        test_dataset=TEST_DATASET,
        model_name=MODEL_NAME,
        model_size=MODEL_SIZE,
    )


