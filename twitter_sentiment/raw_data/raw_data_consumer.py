import json
import logging

from kafka import KafkaConsumer

log = logging.getLogger("root")


class TweetConsumer:
    def __init__(self, topic_name: str) -> None:
        self.consumer = KafkaConsumer(
            topic_name,
            api_version=(0, 10, 2),
            bootstrap_servers=["kafka:9092"],
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="my_group",
            auto_commit_interval_ms=5000,
            fetch_max_bytes=128,
            max_poll_records=100,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        )

    def consume_tweets(self):
        for message in self.consumer:
            tweets = json.loads(json.dumps(message.value))
            tweets = tweets["data"]["text"]
            print(tweets)
