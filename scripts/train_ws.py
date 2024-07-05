import argparse
import json

import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from snorkel.labeling.model import LabelModel

SEED = 42
pl.seed_everything(SEED, workers=True)


def get_train_test_fold(dataset):
    dataset_path = f"data/signals/{dataset}.csv"
    df = pd.read_csv(dataset_path)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    splits = []
    for j, (train_idxs, test_idxs) in enumerate(skf.split(range(len(df)), y=df["objective_true"].to_numpy())):
        train_df, test_df = df.iloc[train_idxs], df.iloc[test_idxs]
        splits.append((train_df, test_df))

    return splits


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    return args


def main(dataset):
    kfold_splits = get_train_test_fold(dataset)

    metrics = []
    for (df_train, df_test), fold in zip(kfold_splits, range(10)):
        print(f"Fold {fold}")
        y_test_gold = df_test["objective_true"].to_numpy()

        # Train label model and infer test labels
        L_ws_train = df_train.iloc[:, :19].to_numpy()
        L_ws_test = df_test.iloc[:, :19].to_numpy()
        label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
        label_model.fit(L_ws_train, n_epochs=500, seed=SEED)
        y_pred_ws = label_model.predict(L=L_ws_test, tie_break_policy="random")
        val_acc = accuracy_score(y_test_gold, y_pred_ws)
        val_f1_macro = f1_score(y_test_gold, y_pred_ws, average="macro")

        tn, fp, fn, tp = confusion_matrix(y_test_gold, y_pred_ws).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        precision = precision_score(y_test_gold, y_pred_ws)
        recall = recall_score(y_test_gold, y_pred_ws)

        metrics.append(
            {
                "acc": val_acc,
                "f1_macro": val_f1_macro,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "true_negative_rate": true_negative_rate,
                "false_negative_rate": false_negative_rate,
                "precision": precision,
                "recall": recall,
            }
        )

    # Average metrics
    avg_metrics = {}
    std_metrics = {}
    for key in metrics[0].keys():
        avg_metrics[key] = sum([m[key] for m in metrics]) / len(metrics)
        std_metrics[key] = sum([(m[key] - avg_metrics[key]) ** 2 for m in metrics]) / len(metrics)

    # Combine avg and std for each metric
    combined_metrics = {}
    for key in avg_metrics.keys():
        metric_mean = avg_metrics[key]
        metric_std = std_metrics[key]
        metric_combined = f"{round(metric_mean, 2)} | {round(metric_std, 2)}"
        combined_metrics[key] = metric_combined

    # Save metrics to json file
    with open(f"{dataset}_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    DATASET = args.dataset

    main(dataset=DATASET)
