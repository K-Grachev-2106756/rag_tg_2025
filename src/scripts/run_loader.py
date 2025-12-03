from data.client import pyro_source


import argparse
import pandas as pd
from structlog import get_logger
import os


BATCH_SIZE = 256
logger = get_logger()


def save_batch(df: pd.DataFrame, out_path: str, is_first_batch: bool):
    if is_first_batch:
        df.to_csv(out_path, index=False, mode="w")
    else:
        df.to_csv(out_path, index=False, mode="a", header=False)


def main():
    parser = argparse.ArgumentParser(description="Telegram posts loader")

    parser.add_argument("--channel_id", type=str, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()
    total_limit = args.limit
    channel_id = args.channel_id
    base_offset = args.offset
    

    out_path = f"./channel_{channel_id}_posts.csv"
    is_first_batch = not os.path.exists(out_path)


    total_batches = (total_limit + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        logger.info(f"Batch #{batch_num} loading")

        current_offset = base_offset + batch_num * BATCH_SIZE

        posts = pyro_source.load_messages(
            channel_id=channel_id,
            limit=BATCH_SIZE,
            offset=current_offset
        )

        df = pd.DataFrame(posts)
        save_batch(df, out_path, is_first_batch)
        is_first_batch = False


    logger.info("Finished loading")

if __name__ == "__main__":
    main()
