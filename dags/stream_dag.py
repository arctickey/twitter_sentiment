# %%
import logging
import os

import airflow
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from src.config import Config
from src.logger import TweetLogger
from src.raw_data.raw_data_consumer import TweetConsumer
from src.raw_data.raw_data_producer import TweetProducer

# %%

log = logging.getLogger("root")
log.addHandler(TweetLogger())

default_args = {
    "owner": "Filip Chrzuszcz",
    "depends_on_past": False,
    "start_date": airflow.utils.dates.days_ago(0),
    "email": ["filipchrzuszcz1@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}


dag = DAG(
    "tweets_collector",
    description="Collect tweets and preprocess them",
    schedule_interval="@once",
    default_args=default_args,
)
task0 = DummyOperator(task_id="dummy", dag=dag)

producer = TweetProducer()
task1a = PythonOperator(
    task_id="product_tweets",
    python_callable=producer.produce_tweets,
    op_kwargs={
        "bearer_token": os.environ["TWITTER_BEARER_TOKEN"],
        "topic_name": Config.TOPIC_NAME,
        "rule": Config.FILTER_RULE,
    },
    dag=dag,
)
consumer = TweetConsumer(topic_name=Config.TOPIC_NAME)
task1b = PythonOperator(
    task_id="consume_tweets",
    python_callable=consumer.consume_tweets,
    op_kwargs={},
    dag=dag,
)


task0 >> [task1a, task1b]
