import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from model import TransformersClassifier, FakeDataset
from sklearn.model_selection import StratifiedKFold
from snorkel.labeling.model import LabelModel
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import wandb

SEED = 42
pl.seed_everything(SEED, workers=True)

def get_train_test_fold(fold, dataset, model_size, model_name="llama2_platypus", num_splits=10):
    assert fold < num_splits
    
    dataset_path = f"data/processed/{dataset}/{model_name}/{model_size}/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for j, (train_idxs, test_idxs) in enumerate(skf.split(range(len(df)), y=df["objective_true"].to_numpy())):
        train_df, test_df = df.iloc[train_idxs], df.iloc[test_idxs]

        if fold == j:
            return train_df, test_df
        
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model_size", type=int, choices=[7, 13, 70], required=True)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--end_classifier_name", type=str, default="roberta-base")
    parser.add_argument("--training_method", type=str, choices=["ws", "zs"])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()
    
    return args


def main(fold, training_method, dataset, model_size, model_name, pretrained_name, hyperparameters):
    experiment_config = {
        "dataset": dataset,
        "fold": fold,
        "model_size": model_size,
        "model_name": model_name,
        "training_method": training_method
    }

    logger = WandbLogger(
        project="prompted_credibility",
        name=f"{training_method}-{dataset}-{model_size}-fold{fold}",
        log_model=False
    )
    if rank_zero_only.rank == 0:
        logger.experiment.config.update(
            experiment_config
        )

    df_train, df_test = get_train_test_fold(fold, dataset, model_size, model_name)
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

        # Train label model and infer silver labels to fine-tune a classifier (label model + end classifier approach)
        L_ws = df_train.iloc[:, :19].to_numpy()
        label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
        label_model.fit(L_ws, n_epochs=500, seed=SEED)
        y_train = label_model.predict(L=L_ws, tie_break_policy="random")

    elif training_method == "zs":
        # Score LLM zero-shot on test set
        zs_test_pred = df_test["objective_pred"].to_numpy()

        val_acc = accuracy_score(y_test_gold, zs_test_pred)
        val_f1_macro = f1_score(y_test_gold, zs_test_pred, average='macro')

        tn, fp, fn, tp = confusion_matrix(y_test_gold, zs_test_pred).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        precision = precision_score(y_test_gold, zs_test_pred)
        recall = recall_score(y_test_gold, zs_test_pred)
        # Get LLM zero-shot train labels to train end classifier
        y_train = df_train["objective_pred"].to_numpy()

    else:
        raise Exception("Training method not supported.")

    # Log the outputs directly from the LLM (without training the end classifier)
    wandb.log({
            "llm_output/val_acc": val_acc,
            "llm_output/f1_macro": val_f1_macro,
            "llm_output/val_true_positive_rate": true_positive_rate,
            "llm_output/val_false_positive_rate": false_positive_rate,
            "llm_output/val_true_negative_rate": true_negative_rate,
            "llm_output/val_false_negative_rate": false_negative_rate,
            "llm_output/precision": precision,
            "llm_output/recall": recall
    }, step=0)
    
    # Train the end classifier
    trainset = FakeDataset(X_train, y_train, tokenizer_name=pretrained_name) # y_train are silver labels
    testset = FakeDataset(X_test, y_test_gold, tokenizer_name=pretrained_name) # y_test_gold are gold labels

    train_loader = DataLoader(trainset, batch_size=hyperparameters["batch_size"], shuffle=True)
    val_loader = DataLoader(testset, batch_size=hyperparameters["batch_size"])

    model = TransformersClassifier(
        pretrained_name=pretrained_name,
        num_classes=1,
        learning_rate=hyperparameters["learning_rate"],
        warmup_steps=hyperparameters["warmup_steps"],
        weight_decay=hyperparameters["weight_decay"]
    )

    trainer = pl.Trainer(deterministic=True, max_epochs=hyperparameters["num_epochs"], logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = parse_arguments()
    DATASET = args.dataset
    FOLD = args.fold
    MODEL_SIZE = args.model_size
    MODEL_NAME = args.model_name
    DEVICE_NUM = args.device_num
    PRETRAINED_NAME = args.end_classifier_name
    TRAINING_METHOD = args.training_method

    hyperparameters = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay
    }

    main(
        fold=FOLD,
        training_method=TRAINING_METHOD,
        dataset=DATASET,
        model_name=MODEL_NAME,
        model_size=MODEL_SIZE,
        pretrained_name=PRETRAINED_NAME,
        hyperparameters=hyperparameters
    )


