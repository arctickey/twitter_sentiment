# syntax=docker/dockerfile:1
FROM python:3.10-slim
RUN apt-get update && \
    apt-get install -y git && \
    apt-get -y install curl && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant
RUN export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
WORKDIR /code
COPY poetry.lock pyproject.toml /code/
COPY dags /code/dags/
