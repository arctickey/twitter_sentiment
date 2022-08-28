import tweepy


class Config:
    HASHTAG = "bitcoin"
    TOPIC_NAME = "tweets"
    FILTER_RULE = tweepy.StreamRule("bitcoin lang:en")
    CONNECTION_STRING_RAW = "postgresql://admin:admin@localhost:5432/tweets_raw"
    MAX_TWEET_LENGTH = 280
    RAW_TABLE_NAME = "raw_processed"
    PROCESSED_TABLE_NAME = "stream_processed"
    PREDICTION_TABLE_NAME = "stream_predicted"
    TRAIN_TABLE_NAME = "train_processed"
    RETRAIN_MODEL = False
