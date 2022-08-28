import pandas as pd
import psycopg2 as pg


def read_from_postgress(table: str) -> pd.DataFrame:
    engine = pg.connect("dbname='db' user='admin' host='db' port='5432' password='admin'")
    df = pd.read_sql(f"select * from {table}", con=engine)
    return df


def save_to_postgress(df: pd.DataFrame, table: str, if_exists: str) -> None:
    engine = pg.connect("dbname='db' user='admin' host='db' port='5432' password='admin'")
    df.to_sql(table, engine, if_exists=if_exists, index=False)
