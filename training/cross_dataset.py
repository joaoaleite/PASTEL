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

    logger = WandbLogger(
        project="prompted_credibility_cross_dataset",
        name=f"{training_method}-TRAIN:{train_dataset}-TEST:{test_dataset}",
        log_model=False
    )

    if rank_zero_only.rank == 0:
        logger.experiment.config.update(
            experiment_config
        )

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


