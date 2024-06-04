import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def encode_sentence_transformer(articles, multiprocess=False):
    if multiprocess is True:
        pool = model.start_multi_process_pool()
        embeddings = model.encode_multi_process(articles, pool, batch_size=128)
        model.stop_multi_process_pool(pool)
    else:
        embeddings = model.encode(articles, batch_size=256)
    return embeddings


def index_by_date_and_sort_index(df, date_field_name="date"):
    if date_field_name not in df:
        raise ValueError(f"Field {date_field_name} not in df.")
    df.loc[:, date_field_name] = pd.to_datetime(
        df[date_field_name], infer_datetime_format=True
    )
    df = df.set_index(date_field_name, drop=True)
    df = df.sort_index()
    return df


def sliding_window(df, window_days=5, min_window_size=100):
    time_min = df["date_publish"].min()
    time_max = df["date_publish"].max()
    logger.info(f"Time min: {time_min}")
    logger.info(f"Time max: {time_max}")

    windows = []

    total_days = (time_max - time_min).days + 1

    logger.info(f"Num days: {total_days}")

    # starting position
    window_left = time_min
    window_right = time_min + pd.Timedelta(days=window_days)

    for i in range(total_days):
        indices = np.where(
            (df["date_publish"] < window_right) & (df["date_publish"] >= window_left)
        )[0]
        if len(indices) >= min_window_size:
            windows.append(df.index[indices])

        window_left += pd.Timedelta(days=1)
        window_right += pd.Timedelta(days=1)

    logger.info(f"Num windows: {len(windows)}")

    return windows
