import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import wandb
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import TransformersClassifier, FakeDataset
from sklearn.model_selection import StratifiedKFold
from snorkel.labeling.model import LabelModel
import argparse


pl.seed_everything(42)

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
    
    
    args = parser.parse_args()
    
    return args


def main(fold, dataset, model_size, model_name):
    df_train, df_test = get_train_test_fold(0, dataset, model_size, model_name)

    # Train label model and infer silver labels
    L_ws = df_train.iloc[:, :19].to_numpy()
    label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
    label_model.fit(L_ws, n_epochs=500, seed=0)
    y_pred_ws = label_model.predict(L=L_ws, tie_break_policy="random")

    # Train with silver labels, evaluate with gold labels
    X_train = df_train["text"].tolist()
    y_train = y_pred_ws

    X_test = df_test["text"].tolist()
    y_test = df_test["objective_true"].to_numpy()

    trainset = FakeDataset(X_train, y_train)
    testset = FakeDataset(X_test, y_test)

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader = DataLoader(testset, batch_size=32)

    model = TransformersClassifier(num_classes=1)
    # wandb.init(project='prompted_credibility', config=model.hparams)
    trainer = pl.Trainer()
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = parse_arguments()
    DATASET = args.dataset
    FOLD = args.dataset
    MODEL_SIZE = args.model_size
    MODEL_NAME = args.model_name
    DEVICE_NUM = args.device_num
    
    main(fold=FOLD, dataset=DATASET, model_name=MODEL_NAME, model_size=MODEL_SIZE)


