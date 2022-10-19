from datetime import date
from typing import Any, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.config import Config
from src.utils import read_from_postgress
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def compute_metrics(predictions: tuple[np.ndarray, np.ndarray]) -> dict[str, int]:
    """Compute metrics for BERT model"""
    pred, labels = predictions
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


class Dataset(torch.utils.data.Dataset):
    """Create Dataset object needed to train BERT model"""

    def __init__(self, encodings: dict, labels: Union[None, np.ndarray] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[Any, Any]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def create_train_val_set(tokenizer: BertTokenizer) -> tuple[Dataset, Dataset]:
    """Split external csv file into train and test set"""
    df = read_from_postgress(Config.TRAIN_TABLE_NAME)
    X = list(df["text"])
    y = list(df["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=Config.MAX_TWEET_LENGTH)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=Config.MAX_TWEET_LENGTH)
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)
    return train_dataset, val_dataset


args = TrainingArguments(
    output_dir="output",
    disable_tqdm=False,
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)


def train_model_and_save(tokenizer: BertTokenizer, model: BertForSequenceClassification) -> None:
    """Perform training of the model and save it to /models directory"""
    train_dataset, val_dataset = create_train_val_set(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(f"models/sentiment_model_{date.today()}")
