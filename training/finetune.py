# Adapted from https://github.com/mshumer/gpt-llm-trainer and https://github.com/arielnlee/Platypus/blob/main/finetune.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, set_peft_model_state_dict
from trl import SFTTrainer
import random
import argparse
import wandb
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Define the arguments
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--model_size", type=int, choices=[7, 13, 70], required=True)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args


def get_train_test_fold(fold, dataset, num_splits=10):
    assert fold < num_splits

    system_context = """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Ensure that your answers are grounded in reality, truthful and reliable. You must answer the following question: "Does this article contain misinformation?" (Yes/No)"""
    prompt = "{system_context}\n\n### Input:\n{text}\n\n### Response:\n{label}"
    SEED = 42

    dataset_path = f"../data/datasets/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    df = df.fillna("")
    df["prompt"] = df.apply(lambda x: prompt.format(text=(x["title"]+"\n"+x["text"]).strip(), system_context=system_context, label="Yes" if x["objective"] == 1 else "No"), axis=1)
    df = df[["title", "text", "prompt", "objective"]]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for j, (train_idxs, test_idxs) in enumerate(skf.split(range(len(df)), y=df["objective"].to_numpy())):
        train_df, test_df = df.iloc[train_idxs], df.iloc[test_idxs]

        if fold == j:
            return train_df, test_df

class llama2_platypus():
    def __init__(self, model, model_name):
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, add_bos_token=True)
        self.model = model
        self.model = self.model.eval()
        self.device = self.model.device

    def prompt(self, user_message, system_context):
        user_message = self.tokenizer.decode(self.tokenizer.encode(user_message, return_tensors='pt', truncation=True, max_length=3800, add_special_tokens=False)[0]) # ensure the text will fit 4096 tokens
        system_context = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{system_context}"
        prompt = f"{system_context}\n\n### Input:\n{user_message.strip()}\n\n### Response:\n"
        size_prompt = len(self.tokenizer.encode(prompt, return_tensors="pt"))
        while size_prompt >= 4096:
            user_message = user_message[500:]
            prompt = f"{system_context}\n\n### Input:\n{user_message.strip()}\n\n### Response:\n"
            size_prompt = len(self.tokenizer.encode(prompt, return_tensors="pt"))

        # ans1 = self.get_next_word_probs(prompt, allow_abstain)
        ans = self.model.generate(self.tokenizer.encode(prompt, return_tensors='pt').to(self.device), max_new_tokens=1)
        ans = self.tokenizer.decode(ans[0])
        ans = ans.split("### Response:")[1].strip()
        return ans
    
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

# %%
if __name__ == "__main__":
    args = parse_arguments()
    DATASET = args.dataset
    FOLD = args.fold
    MODEL_SIZE = args.model_size
    MODEL_NAME = args.model_name

    experiment_config = {
        "dataset": DATASET,
        "fold": FOLD,
        "model_size": MODEL_SIZE,
        "model_name": MODEL_NAME
    }

    model_name = f"garage-bAInd/Platypus2-{MODEL_SIZE}B"
    lora_r = 16
    lora_alpha = 16
    lora_dropout= 0.05
    # use_4bit = True
    # bnb_4bit_compute_dtype = "float16"
    # bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    output_dir = "./results"
    num_train_epochs = 1
    fp16 = False
    bf16 = False
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 3e-4
    weight_decay = 0.000
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 25
    logging_steps = 5
    max_seq_length = None
    packing = False
    device_map = {"": 0}
    train_df, test_df = get_train_test_fold(FOLD, DATASET)

    wandb.init(project="test-llm-train", name=f"{DATASET}-{MODEL_SIZE}-fold{FOLD}")
    wandb.config["experiment"] = experiment_config
    # Load datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(test_df)

    # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # quantization_config=bnb_config,
        load_in_8bit=True,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="all",
        evaluation_strategy="steps",
        eval_steps=5  # Evaluate every 20 steps
    )
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  # Pass validation dataset here
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.train()
    # trainer.model.save_pretrained(new_model)

    print("Making inference...")
    inference_model = llama2_platypus(model_name=model_name, model=model)
    system_context = """You are a helpful and unbiased news verification assistant. You will be provided with the title and the full body of text of a news article. Ensure that your answers are grounded in reality, truthful and reliable. You must answer the following question: "Does this article contain misinformation?" (Yes/No)"""
    system_context_zs = system_context.format(options="Yes/No", abstain_context="", question="Does this article contain misinformation?")
    prompt = """{title}\n{text}"""
    randomizer = lambda: random.randint(0, 1)
    def class_mapper(x):
        if x.lower().startswith("yes"):
            return 1
        elif x.lower().startswith("no"):
            return 0
        else:
            return -1

    preds = []
    targets = []
    for i, article_row in enumerate(test_df.sample(frac=1).itertuples()):
        prompt_formatted = prompt.format(title=article_row.title, text=article_row.text, system_context=system_context_zs)
        ans = inference_model.prompt(prompt_formatted, system_context=system_context_zs)
        label = class_mapper(ans)

        preds.append(label)
        targets.append(article_row.objective)

    num_invalid = len([v for v in preds if v == -1])
    percent_invalid = num_invalid/len(targets)
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

    wandb.log({
        'task_test/acc': val_acc,
        'task_test/f1_macro': val_f1_macro,
        'task_test/false_positive_rate': false_positive_rate,
        'task_test/true_negative_rate': true_negative_rate,
        'task_test/false_negative_rate': false_negative_rate,
        'task_test/true_positive_rate': true_positive_rate,
        'task_test/precision': precision,
        'task_test/recall': recall,
        'task_test/num_invalid': num_invalid,
        'task_test/percent_invalid': percent_invalid
    }, step=wandb.run.step)


