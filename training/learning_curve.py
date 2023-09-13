import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from model import TransformersClassifier, FakeDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from snorkel.labeling.model import LabelModel
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import wandb

SEED = 42
pl.seed_everything(SEED, workers=True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model_size", type=int, choices=[7, 13, 70], required=True)
    parser.add_argument("--model_name", type=str, default="llama2_platypus")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--end_classifier_name", type=str, default="roberta-base")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()
    
    return args

def get_train_test_split(step, dataset, model_name, model_size):
    dataset_path = f"../data/processed/{dataset}/{model_name}/{model_size}/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    train_df, test_df = train_test_split(df, train_size=0.7, random_state=SEED)

    train_df_subset = train_df.iloc[:int(len(train_df)*step//10)] # get {step*10}% of the train data

    return train_df_subset, test_df

def get_train_test_fold(fold, step, dataset, model_size, model_name="llama2_platypus", num_splits=10):
    assert fold < num_splits
    
    dataset_path = f"../data/processed/{dataset}/{model_name}/{model_size}/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for j, (train_idxs, test_idxs) in enumerate(skf.split(range(len(df)), y=df["objective_true"].to_numpy())):
        train_df, test_df = df.iloc[train_idxs], df.iloc[test_idxs]
        train_df_subset = train_df.iloc[:int(len(train_df)*step//10)] # get (step*10)% of the train set

        if fold == j:
            return train_df_subset, test_df

def main(fold, dataset, model_size, model_name, pretrained_name, hyperparameters):
    experiment_config = {
        "dataset": dataset,
        "model_size": model_size,
        "model_name": model_name,
        "fold": fold
    }

    logger = WandbLogger(
        project="prompted_credibility-learning_curve",
        name=f"{dataset}-{model_name}-{model_size}-fold{fold}",
        log_model=False
    )
    if rank_zero_only.rank == 0:
        logger.experiment.config.update(
            experiment_config
        )

    # Will train and evaluate models with increasing steps of +10% of the train set from 10% to 100%.
    for step in range(1, 11):
        df_train, df_test = get_train_test_fold(
            fold=fold, step=step, dataset=dataset, model_name=model_name, model_size=model_size)
        X_train = df_train["text"].tolist()
        X_test = df_test["text"].tolist()
        y_test_gold = df_test["objective_true"].to_numpy()


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

        # Train label model and infer silver labels to fine-tune a classifier (label model + end classifier approach)
        L_ws = df_train.iloc[:, :19].to_numpy()
        label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
        label_model.fit(L_ws, n_epochs=500, seed=SEED)
        y_train = label_model.predict(L=L_ws, tie_break_policy="random")

        wandb.log({
                "weak_supervision/val_acc": val_acc,
                "weak_supervision/f1_macro": val_f1_macro,
                "weak_supervision/val_true_positive_rate": true_positive_rate,
                "weak_supervision/val_false_positive_rate": false_positive_rate,
                "weak_supervision/val_true_negative_rate": true_negative_rate,
                "weak_supervision/val_false_negative_rate": false_negative_rate,
                "weak_supervision/precision": precision,
                "weak_supervision/recall": recall
        }, step=step*10)
    
        # Score LLM zero-shot on test set
        zs_test_pred = df_test["objective_pred"].to_numpy()

        val_acc = accuracy_score(y_test_gold, zs_test_pred)
        val_f1_macro = f1_score(y_test_gold, zs_test_pred, average='macro', zero_division=0.0)

        tn, fp, fn, tp = confusion_matrix(y_test_gold, zs_test_pred).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        precision = precision_score(y_test_gold, zs_test_pred, zero_division=0.0)
        recall = recall_score(y_test_gold, zs_test_pred, zero_division=0.0)
        # Get LLM zero-shot train labels to train end classifier
        y_train = df_train["objective_pred"].to_numpy()

        wandb.log({
                "zero_shot/val_acc": val_acc,
                "zero_shot/f1_macro": val_f1_macro,
                "zero_shot/val_true_positive_rate": true_positive_rate,
                "zero_shot/val_false_positive_rate": false_positive_rate,
                "zero_shot/val_true_negative_rate": true_negative_rate,
                "zero_shot/val_false_negative_rate": false_negative_rate,
                "zero_shot/precision": precision,
                "zero_shot/recall": recall
        }, step=step*10)

        # Fine-tune classifier with gold labels
        y_train = df_train["objective_true"].to_numpy()
        
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

        trainer = pl.Trainer(deterministic=True, max_epochs=hyperparameters["num_epochs"], logger=None)
        trainer.fit(model, train_loader, val_loader)

        # Log the scores of the best epoch
        gl_best_scores = model.best_scores
        gl_best_scores = {f"gl/{k}":v for k, v in gl_best_scores.items()}

        wandb.log(gl_best_scores, step=step*10)

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
    dataset=DATASET,
    model_name=MODEL_NAME,
    model_size=MODEL_SIZE,
    pretrained_name=PRETRAINED_NAME,
    hyperparameters=hyperparameters
)