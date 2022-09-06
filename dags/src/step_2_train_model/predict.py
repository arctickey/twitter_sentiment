import numpy as np
from src.config import Config
from src.step_2_train_model.train_model import Dataset
from src.utils import read_from_postgress, save_to_postgress
from transformers import BertForSequenceClassification, BertTokenizer, Trainer


def predict(model_path: str, tokenizer: BertTokenizer) -> None:
    test = read_from_postgress(Config.PROCESSED_TABLE_NAME)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    X_test = list(test["text"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=Config.MAX_TWEET_LENGTH)
    test_dataset = Dataset(X_test_tokenized)
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    y_pred = np.argmax(raw_pred, axis=1)
    test["sentiment"] = y_pred
    save_to_postgress(test, Config.PROCESSED_TABLE_NAME, if_exists="append")
