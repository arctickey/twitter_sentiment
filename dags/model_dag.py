# %%
import logging
from ast import literal_eval

import airflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from src.logger import TweetLogger
from src.step_1_preprocess_data.preprocess_data import (
    preprocess_stream,
    preprocess_train,
)
from src.step_2_train_model.predict import predict
from src.step_2_train_model.train_model import train_model_and_save
from transformers import BertForSequenceClassification, BertTokenizer

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


model_path = "./models/bert-tiny/"
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2, local_files_only=True)


dag = DAG(
    "tweets_sentiment_predictor",
    description="Preprocess tweets and determine sentiment",
    schedule_interval="*/30 * * * *",
    default_args=default_args,
)


def branch() -> str:
    if literal_eval(Variable.get("retrain_model")):
        return "preprocess_train"
    else:
        return "predict_stream"


branch_task = BranchPythonOperator(task_id="branch_task", python_callable=branch, trigger_rule="all_done", dag=dag)

task2a = PythonOperator(
    task_id="preprocess_stream",
    python_callable=preprocess_stream,
    dag=dag,
)
task2b = PythonOperator(
    task_id="preprocess_train",
    python_callable=preprocess_train,
    op_kwargs={"path": "./db/marked_tweets.csv"},
    dag=dag,
)


task2c = PythonOperator(
    task_id="train_model",
    python_callable=train_model_and_save,
    op_kwargs={"tokenizer": tokenizer, "model": model},
    dag=dag,
)
task2e = PythonOperator(
    task_id="predict_stream",
    python_callable=predict,
    op_kwargs={"model_path": model_path, "tokenizer": tokenizer},
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)
task2a >> branch_task
branch_task >> task2b >> task2c >> task2e
branch_task >> task2e
