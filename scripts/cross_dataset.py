import argparse
import json

import evaluate
import pandas as pd
import pytorch_lightning as pl
from datasets import Dataset
from sklearn.metrics import f1_score
from snorkel.labeling.model import LabelModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
pl.seed_everything(SEED, workers=True)


def get_train_test(train_dataset, test_dataset):
    dataset_path = f"data/signals/{train_dataset}.csv"
    train_df = pd.read_csv(dataset_path)

    dataset_path = f"data/signals/{test_dataset}.csv"
    test_df = pd.read_csv(dataset_path)

    return train_df, test_df


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--training_method", type=str, choices=["ws", "ft"])
    args = parser.parse_args()

    return args


def run_pastel(df_train, df_test):

    y_test_gold = df_test["objective_true"].to_numpy()
    L_ws_train = df_train.iloc[:, :19].to_numpy()
    L_ws_test = df_test.iloc[:, :19].to_numpy()
    label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
    label_model.fit(L_ws_train, n_epochs=500, seed=SEED)
    y_pred_ws = label_model.predict(L=L_ws_test, tie_break_policy="random")
    val_f1_macro = f1_score(y_test_gold, y_pred_ws, average="macro")

    return val_f1_macro


def run_roberta(train_df, test_df):
    PRETRAINED_NAME = "roberta-base"

    metric = evaluate.load("f1")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    config = {"batch_size": 8, "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_steps": 0, "train_epochs": 5}

    train_df = train_df[["title", "text", "objective"]]
    train_df = train_df.fillna("")
    train_df = train_df[train_df["text"] != ""]
    train_df["label"] = train_df["objective"]
    train_df["text"] = train_df["title"] + "\n" + train_df["text"]

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    # define the model
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_NAME, num_labels=2)

    training_args = TrainingArguments(
        evaluation_strategy="no",
        output_dir=".",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["train_epochs"],
        weight_decay=config["weight_decay"],
        push_to_hub=False,
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
    )

    # train the model
    trainer.train()

    test_df = test_df[["title", "text", "objective"]]
    test_df = test_df.fillna("")
    test_df = test_df[test_df["text"] != ""]
    test_df["label"] = test_df["objective"]
    test_df["text"] = test_df["title"] + "\n" + test_df["text"]

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # predict
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    f1_score = metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    return f1_score


def main(train_dataset, test_dataset):
    df_train, df_test = get_train_test(train_dataset=train_dataset, test_dataset=test_dataset)

    pastel_f1 = run_pastel(df_train, df_test)
    roberta_f1 = run_roberta(df_train, df_test)

    metrics = {
        "pastel_f1": pastel_f1,
        "roberta_f1": roberta_f1,
    }

    with open(f"results_train_{train_dataset}_test{test_dataset}.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    TRAIN_DATASET = args.train_dataset
    TEST_DATASET = args.test_dataset

    main(train_dataset=TRAIN_DATASET, test_dataset=TEST_DATASET)
