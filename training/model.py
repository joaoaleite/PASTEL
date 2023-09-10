import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class FakeDataset(Dataset):
    def __init__(self, data, targets, max_len=512):
        self.data = data
        self.targets = targets
        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
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
    def __init__(self, num_classes, learning_rate=2e-5):
        super(TransformersClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, targets=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = self(input_ids, attention_mask, targets)
        loss = outputs.loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch.values()

        outputs = self(input_ids, attention_mask, targets)
        val_loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        val_acc = accuracy_score(targets.cpu(), preds.cpu())
        val_f1_macro = f1_score(targets.cpu(), preds.cpu(), average='macro')

        tn, fp, fn, tp = confusion_matrix(targets.cpu(), preds.cpu()).ravel()
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1_macro', val_f1_macro, prog_bar=True)
        self.log('val_false_positive_rate', false_positive_rate, prog_bar=True)
        self.log('val_true_negative_rate', true_negative_rate, prog_bar=True)
        self.log('val_false_negative_rate', false_negative_rate, prog_bar=True)
        self.log('val_true_positive_rate', true_positive_rate, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer