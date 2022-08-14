import tweepy


class Config:
    HASHTAG = "bitcoin"
    TOPIC_NAME = "tweet"
    FILTER_RULE = tweepy.StreamRule("bitcoin lang:en")
