import logging
from typing import Any

import tweepy
from kafka import KafkaProducer

log = logging.getLogger("root")


class MyStreamListener(tweepy.StreamingClient):
    def __init__(self, bearer_token: str, topic_name: str) -> None:
        super().__init__(bearer_token)
        self.producer = KafkaProducer(bootstrap_servers="kafka:9092", api_version=(0, 10, 2))
        self.topic_name = topic_name

    def on_data(self, data: Any) -> None:
        self.producer.send(self.topic_name, data)

    def on_errors(self, status: Any) -> None:
        log.exception(status)
        raise ValueError(status)


class TweetProducer:
    def produce_tweets(self, bearer_token: str, topic_name: str, rule: tweepy.StreamRule) -> None:
        stream = MyStreamListener(bearer_token=bearer_token, topic_name=topic_name)
        stream.add_rules(rule)
        stream.filter()
