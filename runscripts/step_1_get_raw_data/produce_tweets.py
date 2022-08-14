#  %%
import logging
import os

from twitter_sentiment.confg import Config
from twitter_sentiment.logger import TweetLogger
from twitter_sentiment.raw_data.raw_data_producer import TweetProducer

log = logging.getLogger("root")
log.addHandler(TweetLogger())


if __name__ == "__main__":
    stream = TweetProducer()
    stream.produce_tweets(
        bearer_token=os.environ["TWITTER_BEARER_TOKEN"], topic_name=Config.TOPIC_NAME, rule=Config.FILTER_RULE
    )

# %%
