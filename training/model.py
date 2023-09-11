import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import wandb
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class FakeDataset(Dataset):
    def __init__(self, data, targets, tokenizer_name, max_len=512):
        self.data = data
        self.targets = targets
        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

class TransformersClassifier(pl.LightningModule):
    def __init__(
            self,
            pretrained_name,
            learning_rate,
            warmup_steps,
            weight_decay,
            num_classes=1):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_classes)
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, targets=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = self(input_ids, attention_mask, targets)
        loss = outputs.loss

        self.log('end_classifier/train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch.values()
        outputs = self(input_ids, attention_mask, targets)
        val_loss = outputs.loss
        logits = outputs.logits
        probs = torch.nn.Sigmoid()(logits)
        preds = torch.zeros(probs.shape)
        preds[torch.where(probs >= 0.5)] = 1
        outputs = {"loss": val_loss, "preds": preds, "labels": targets}

        self.validation_step_outputs.append(outputs)

        return outputs

    def on_validation_epoch_end(self):
        if self.global_step == 0:
            wandb.define_metric('end_classifier/val_f1_macro', summary='max')
            wandb.define_metric('end_classifier/val_acc', summary='max')
            wandb.define_metric('end_classifier/val_loss', summary='min')
            wandb.define_metric('end_classifier/val_false_positive_rate', summary='min')
            wandb.define_metric('end_classifier/val_true_negative_rate', summary='min')
            wandb.define_metric('end_classifier/val_false_negative_rate', summary='min')
            wandb.define_metric('end_classifier/val_true_positive_rate', summary='min')
            
    
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        targets = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        val_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()


        val_acc = accuracy_score(targets, preds)
        val_f1_macro = f1_score(targets, preds, average='macro', zero_division=0.0)

        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        self.log('end_classifier/val_loss', val_loss, prog_bar=True)
        self.log('end_classifier/val_acc', val_acc, prog_bar=True)
        self.log('end_classifier/val_f1_macro', val_f1_macro, prog_bar=True)
        self.log('end_classifier/val_false_positive_rate', false_positive_rate, prog_bar=True)
        self.log('end_classifier/val_true_negative_rate', true_negative_rate, prog_bar=True)
        self.log('end_classifier/val_false_negative_rate', false_negative_rate, prog_bar=True)
        self.log('end_classifier/val_true_positive_rate', true_positive_rate, prog_bar=True)

        print(classification_report(targets, preds))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]