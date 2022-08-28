# %%
import logging

import airflow
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from src.config import Config
from src.logger import TweetLogger
from src.step_1_preprocess_data.preprocess_data import (
    preprocess_stream,
    preprocess_train,
)
from src.step_2_train_model.predict import predict
from src.step_2_train_model.train_model import train_model_and_save

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
    "tweets sentiment predictor",
    description="Preprocess tweets and determine sentiment",
    schedule_interval="*/10 * * * *",
    default_args=default_args,
)


def branch():
    if Config.RETRAIN_MODEL:
        return "preprocess_train"
    else:
        return "predict_stream"


branch_task = BranchPythonOperator(task_id="branch", python_callable=branch, trigger_rule="all_done", dag=dag)

task2a = PythonOperator(
    task_id="preprocess_stream",
    python_callable=preprocess_stream,
    op_kwargs={"path", "db/marked_tweets.csv"},
    dag=dag,
)
task2b = PythonOperator(
    task_id="preprocess_train",
    python_callable=preprocess_train,
    dag=dag,
)


task2c = PythonOperator(
    task_id="train_model",
    python_callable=train_model_and_save,
    op_kwargs={"path", "db/marked_tweets.csv"},
    dag=dag,
)
task2d = DummyOperator(task_id="merger", trigger_rule=TriggerRule.ONE_SUCCESS, dag=dag)

task2e = PythonOperator(
    task_id="predict_stream",
    python_callable=predict,
    op_kwargs={"model_path", "models/sentiment_model_2022-08-28"},
    dag=dag,
)
task2a >> branch_task
branch_task >> task2b >> task2c >> task2d
branch_task >> task2d
task2d >> task2e
