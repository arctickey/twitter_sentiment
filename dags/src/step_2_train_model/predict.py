import logging

import numpy as np
from src.config import Config
from src.logger import TweetLogger
from src.step_2_train_model.train_model import Dataset
from src.utils import db_table_exists, read_from_postgress, save_to_postgress
from transformers import BertForSequenceClassification, BertTokenizer, Trainer

log = logging.getLogger("root")
log.addHandler(TweetLogger())


def predict(model_path: str, tokenizer: BertTokenizer) -> None:
    """Predict sentiment of new tweets and save it to database"""
    df_processed = read_from_postgress(Config.PROCESSED_TABLE_NAME)
    if db_table_exists(table=Config.OUTPUT_TABLE_NAME):
        df_already_predicted = read_from_postgress(table=Config.PROCESSED_TABLE_NAME)
        df_sentiment = df_processed.merge(
            df_already_predicted, on="id", how="left", indicator=True, suffixes=("", "_y")
        )
        df_sentiment = df_sentiment.loc[df_sentiment["_merge"] == "left_only", ["text", "id", "timestamp"]]
    else:
        log.info("Sentiment table does not exist, creating new.")
        df_sentiment = df_processed
    log.info("Data ready to predict")
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    X_test = list(df_sentiment["text"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=Config.MAX_TWEET_LENGTH)
    test_dataset = Dataset(X_test_tokenized)
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    y_pred = np.argmax(raw_pred, axis=1)
    df_sentiment["sentiment"] = y_pred
    save_to_postgress(df_sentiment, Config.OUTPUT_TABLE_NAME, if_exists="append")
