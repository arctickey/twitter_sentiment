import tweepy


class Config:
    HASHTAG = "bitcoin"
    TOPIC_NAME = "tweets"
    FILTER_RULE = tweepy.StreamRule("bitcoin lang:en")
    CONNECTION_STRING_RAW = "postgresql://admin:admin@localhost:5432/tweets_raw"
