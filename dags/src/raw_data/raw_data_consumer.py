import json
import logging

import pyspark.sql.functions as F
from kafka import KafkaConsumer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

log = logging.getLogger("root").setLevel(logging.ERROR)


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
        self.spark = (
            SparkSession.builder.appName("tweets_analyze")
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,org.postgresql:postgresql:42.2.18",
            )
            .getOrCreate()
        )

    def consume_tweets(self) -> None:
        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", "kafka:9092")
            .option("subscribe", "tweets")
            .option("startingOffsets", "earliest")
            .load()
        )
        schema = StructType([StructField("text", StringType(), True)])
        schema = StructType(
            [
                StructField("data", schema, True),
            ]
        )
        values = df.select([F.from_json(df.value.cast("string"), schema).alias("tweet"), "timestamp"])
        df_out = values.select(*["tweet.data.text", "timestamp"])
        df_cleaned = self.clean_tweets(df_out)
        query = df_cleaned.writeStream.queryName("append_tweets").foreachBatch(self._append_to_db).start()
        query.awaitTermination()

    def clean_tweets(self, tweets: DataFrame) -> DataFrame:
        strings_to_sub = ["http\S+", "bit\.ly\S+", "(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)", "(@[A-Za-z]+[A-Za-z0-9-_]+)"]
        tweets = tweets.withColumn("text", F.regexp_replace("text", "|".join(strings_to_sub), ""))
        tweets = tweets.withColumn("text", F.trim(F.col("text")))
        return tweets

    def _append_to_db(self, df: DataFrame, epoch_id: int) -> None:
        df.write.mode("append").format("jdbc").option("url", "jdbc:postgresql://db:5432/db").option(
            "dbtable", "tweets"
        ).option("user", "admin").option("password", "admin").option("driver", "org.postgresql.Driver").save()
