import os
import sys
sys.path.append(os.getcwd())

import pandas as pd

from src.db_utils.sql_utils import sql_drop, sql_dump_df
from src.data.clean import clean_df




if __name__ == "__main__":
    # Предобработка документов
    rbc = pd.read_csv("src/dataset/rbc/channel_rbc_news_posts.csv")
    rbc = clean_df(rbc)

    # Загрузка в бд
    table = "posts"
    sql_drop(table)
    sql_dump_df(rbc, table, if_exists="replace")
