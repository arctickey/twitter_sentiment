import pandas as pd
from src.config import Config
from src.utils import read_from_postgress, save_to_postgress


def clean_text(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df[text_column] = (
        df.loc[:, text_column]
        .str.replace("\n", "", regex=False)
        .str.strip(": ")
        .str.strip()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace("&amp;", "&")
        .str.replace(r"@([A-Za-z0-9_]+)", "", regex=True)
        .str.replace("\xa0", "", regex=False)
    )
    df = df[df["text"].str.len() <= Config.MAX_TWEET_LENGTH]
    return df


def read_external_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1", header=None)
    df.columns = ["label", "time", "date", "query", "username", "text"]
    df = df.loc[:, ["label", "text"]]
    df["label"] = df["label"] / 4  # change labels to 0 and 1
    return df


def preprocess_stream() -> None:
    df = read_from_postgress(Config.RAW_TABLE_NAME)
    df_already_processed = read_from_postgress(Config.PROCESSED_TABLE_NAME)
    df_to_process = df.merge(df_already_processed, on="id", how="left", indicator=True)
    df_to_process = df_to_process.loc[df_to_process["_merge"] == "left_only", "id"]
    df_to_process = clean_text(df_to_process, "text")
    save_to_postgress(df_to_process, Config.PROCESSED_TABLE_NAME, if_exists="append")
    return


def preprocess_train(path: str) -> None:
    df = read_external_data(path)
    df = clean_text(df, "text")
    save_to_postgress(df, Config.TRAIN_TABLE_NAME, if_exists="replace")
    return
