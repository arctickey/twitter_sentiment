# %%
import logging

from twitter_sentiment.confg import Config
from twitter_sentiment.logger import TweetLogger
from twitter_sentiment.raw_data.raw_data_consumer import TweetConsumer

log = logging.getLogger("root")
log.addHandler(TweetLogger())


if __name__ == "__main__":
    consumer = TweetConsumer(topic_name=Config.TOPIC_NAME)
    consumer.consume_tweets()
