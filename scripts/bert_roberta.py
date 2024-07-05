import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import random
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Define the arguments
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    # Parse the arguments
    args = parser.parse_args()

    return args


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


if __name__ == "__main__":
    args = parse_arguments()
    DATASET = args.dataset
    PRETRAINED_NAME = args.pretrained_model

    if PRETRAINED_NAME == "roberta-base":
        config = {"batch_size": 8, "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_steps": 0, "train_epochs": 5}
    else:
        # Default configs for BERT
        config = {"batch_size": 8, "learning_rate": None, "weight_decay": None, "warmup_steps": None, "train_epochs": 5}

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    final_results = []
    for dataset in ["celebrity", "fakenewsamt", "gossipcop", "politifact"]:
        print(dataset)
        dataset_results = []

        data_path = os.path.join("data", "datasets", f"{dataset}.csv")
        df = pd.read_csv(data_path)
        df = df[["title", "text", "objective"]]
        df = df.fillna("")
        df = df[df["text"] != ""]
        df["label"] = df["objective"]
        df["text"] = df["title"] + "\n" + df["text"]

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for i, (train_idxs, test_idxs) in enumerate(skf.split(df, df["label"])):
            train_df = df.iloc[train_idxs]
            test_df = df.iloc[test_idxs]

            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            test_dataset = test_dataset.map(tokenize_function, batched=True)
            train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            # define the model
            model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_NAME, num_labels=2)

            training_args = TrainingArguments(
                evaluation_strategy="epoch",
                output_dir="models",
                learning_rate=config["learning_rate"],
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["batch_size"],
                num_train_epochs=config["train_epochs"],
                weight_decay=config["weight_decay"],
                push_to_hub=False,
                logging_steps=10,
                load_best_model_at_end=False,
                metric_for_best_model="f1",
                greater_is_better=True,
                save_strategy="no",
                seed=SEED,
                data_seed=SEED,
                warmup_steps=config["warmup_steps"],
            )

            # define the trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics,
            )

            # train the model
            trainer.train()

            # predict
            predictions = trainer.predict(test_dataset)
            preds = predictions.predictions.argmax(-1)
            labels = predictions.label_ids

            f1 = evaluate.load("f1")

            f1_score = f1.compute(predictions=preds, references=labels, average="macro")["f1"]

            print(f"Fold {i} F1: {f1_score}")

            dataset_results.append(
                {
                    "f1": f1_score,
                }
            )

        dataset_results = pd.DataFrame(dataset_results)
        # get mean and std for f1_macro
        final_results.append(
            {"mean": dataset_results.mean()["f1"], "std": dataset_results.std()["f1"], "dataset": dataset}
        )

        pd.DataFrame(final_results).to_csv(os.path.join(".", "results_{PRETRAINED_MODEL}.csv"), index=False)
