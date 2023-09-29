import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from snorkel.labeling.model import LabelModel
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import wandb
from peft import set_peft_model_state_dict, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, AutoTokenizer
from tqdm import tqdm
import random
import os
import torch

SEED = 42
pl.seed_everything(SEED, workers=True)

class llama2_platypus():
    def __init__(self, size, model):
        if size in [7, 13, 70]:
            model_name = f"garage-bAInd/Platypus2-{size}B"
        else:
            raise Exception(f"Size {size} not available for Llama. Choose 7, 13 or 70.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, add_bos_token=True)
        self.model = model

    def prompt(self, input, question, system_context):
        # The number of tokens for the question and prompt formatting amounts to 33 tokens
        # so we can use 4064 tokens for the input text. Will use 4050 to leave some room.
        truncate_to = 4050
        input = self.tokenizer.decode(self.tokenizer.encode(input, return_tensors='pt', truncation=True, max_length=truncate_to, add_special_tokens=False)[0]) # ensure the text will fit 4096 tokens
        prompt = f"### Instruction:\n{system_context}\n\n### Input:\n{input.strip()}\n\n{question.strip()}\n\n### Response:\n"
        # ans1 = self.get_next_word_probs(prompt, allow_abstain)
        genconfig = GenerationConfig(max_new_tokens=1, num_beams=1, do_sample=False)
        inputs = {"inputs": self.tokenizer.encode(prompt, return_tensors='pt').to("cuda"), "generation_config": genconfig}
        ans = self.model.generate(**inputs)
        ans = self.tokenizer.decode(ans[0])
        ans = ans.split("### Response:")[1].strip()
        return ans
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--model_size", type=int, choices=[7, 13, 70], default=70)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--training_method", choices=["ws", "ft"], required=True)

    args = parser.parse_args()
    
    return args

def get_datasets_fraction(dataset, frac, model_name="llama2_platypus"):
    assert frac <= 1.0

    system_context = """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Then, you will answer further questions related to the given article. Ensure that your answers are grounded in reality, truthful and reliable."""
    prompt = "### Instruction:\n{system_context}\n\n### Input:\n{text}\n\n### Response:\n{label}"
    SEED = 42

    dataset_path = f"data/processed/{dataset}/{model_name}/70/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    df["prompt"] = df.apply(lambda x: prompt.format(text=(x["text"]).strip(), system_context=system_context, label="Yes" if x["objective_true"] == 1 else "No"), axis=1)
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=SEED)
    df_train = df_train.sample(frac=frac, random_state=SEED) # Get a fraction of the training set

    return df_train, df_test

def main(dataset, model_size, model_name, fraction, training_method):
    experiment_config = {
        "dataset": dataset,
        "model_size": model_size,
        "model_name": model_name,
        "fraction": fraction,
        "training_method": training_method
    }

    wandb.init(project="prompted_credibility_learning_curve", name=f"{dataset}-{training_method}-FRAC:{fraction}")
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

    # Will train and evaluate models with increasing steps of +10% of the train set from 10% to 100%.
    df_train, df_test = get_datasets_fraction(dataset, fraction)  
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

        precision = precision_score(y_test_gold, y_pred_ws, zero_division=0.0)
        recall = recall_score(y_test_gold, y_pred_ws, zero_division=0.0)

    elif training_method == "ft":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            "garage-bAInd/Platypus2-70B",
            quantization_config=bnb_config,
            device_map="auto"
        )

        lora_r = 8
        lora_alpha = 16
        lora_dropout= 0.05
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model_path = f"finetuning/results-{dataset}-{fraction}/"
        latest_step = sorted([folder.split("-")[1] for folder in os.listdir(model_path) if folder.startswith("checkpoint")], reverse=True).pop()
        best_model_path = os.path.join(model_path, f"checkpoint-{latest_step}", "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)

        print("loading checkpoint:", best_model_path)
        set_peft_model_state_dict(model, adapters_weights)  # set LoRA weights
        inference_model = llama2_platypus(size=MODEL_SIZE, model=model)
        system_context = """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Then, you will answer further questions related to the given article. Ensure that your answers are grounded in reality, truthful and reliable."""
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
        model.eval()
        for i, article_row in tqdm(enumerate(df_test.sample(frac=1).itertuples()), total=len(df_test)):
            system_context_zs = system_context.format(abstain_context="")
            input = article_row.text # already contains title \n text
            question = "Does this article contain misinformation? (Yes/No)"
            ans = inference_model.prompt(input=input, question=question, system_context=system_context_zs)
        
            label = class_mapper(ans)

            preds.append(label)
            targets.append(article_row.objective_true)

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
       
args = parse_arguments()

DATASET = args.dataset
MODEL_SIZE = args.model_size
MODEL_NAME = args.model_name
DEVICE_NUM = args.device_num
FRACTION = args.fraction
TRAINING_METHOD = args.training_method

main(
    dataset=DATASET,
    model_name=MODEL_NAME,
    model_size=MODEL_SIZE,
    fraction=FRACTION,
    training_method=TRAINING_METHOD
)
