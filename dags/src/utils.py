import pandas as pd
import sqlalchemy


def _create_engine() -> sqlalchemy.engine.base.Engine:
    engine = sqlalchemy.create_engine("postgresql://admin:admin@db:5432/db")
    return engine


def read_from_postgress(table: str) -> pd.DataFrame:
    engine = _create_engine()
    df = pd.read_sql(f"select * from {table}", con=engine)
    return df


def save_to_postgress(df: pd.DataFrame, table: str, if_exists: str) -> None:
    engine = _create_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    return


def db_table_exists(table: str) -> bool:
    engine = _create_engine()
    sql = f"select * from information_schema.tables where table_name='{table}'"
    results_df = pd.read_sql(sql, con=engine)
    return bool(len(results_df))
