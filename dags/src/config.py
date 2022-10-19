import tweepy
from airflow.models import Variable


class Config:
    Variable.set(key="retrain_model", value=False)
    HASHTAG = "bitcoin"
    TOPIC_NAME = "tweets"
    FILTER_RULE = tweepy.StreamRule("bitcoin lang:en")
    CONNECTION_STRING_RAW = "postgresql://admin:admin@localhost:5432/tweets_raw"
    MAX_TWEET_LENGTH = 280
    RAW_TABLE_NAME = "raw_data"
    PROCESSED_TABLE_NAME = "stream_processed"
    OUTPUT_TABLE_NAME = "stream_predicted"
    TRAIN_TABLE_NAME = "train_processed"
