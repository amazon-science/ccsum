import os

from ccsum import data_util, text_util, retrieval, filtering
import logging
import pandas as pd
import click
import torch
import hashlib
from datetime import datetime
import gc
import time

logger = logging.getLogger(__name__)


def hash_summary_article(summary, article):
    return hashlib.sha256(f"{summary}[SEP]{article}".encode()).hexdigest()


def create_self_supervised_summaries_from_window_parquet(df, min_cluster_size, input_filename):
    logger.info(f"Processing {len(df)} articles.")
    logger.info(f"Date range: {df['date_publish'].min()} - {df['date_publish'].max()}")

    summary_article_candidates = create_summary_article_pairs_from_one_window(df, df.index.tolist(), min_cluster_size)
    df_paired = summary_article_candidates.to_df()
    df_paired = align_df(df_paired, df)
    df_paired = filtering.apply_filters(df_paired)

    if df_paired is None:
        return None

    df_paired["id"] = (df_paired["summary"] + df_paired["article"]).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    output_filename = f"/home/ec2-user/data/processed_ccnews_summaries/{input_filename.split('/')[-1]}"
    df_paired.to_parquet(output_filename)
    logger.info(f"Dumping processed summaries to {output_filename}")
    gc.collect()
    torch.cuda.empty_cache()


def create_self_supervised_summaries(df, time_window_in_days, min_cluster_size):
    logger.info(f"Processing {len(df)} articles.")
    windows = data_util.sliding_window(df, time_window_in_days)
    for i, w in enumerate(windows):
        logger.info(f"Processing {i}/{len(windows)} sliding windows.")
        logger.info(f"Date range: {df.loc[w]['date_publish'].min()} - {df.loc[w]['date_publish'].max()}")

        time.sleep(5)
        gc.collect()
        torch.cuda.empty_cache()

        df_window = df.loc[w]
        summary_article_candidates = create_summary_article_pairs_from_one_window(df_window, w, min_cluster_size)
        df_paired = summary_article_candidates.to_df()
        df_paired = align_df(df_paired, df)
        df_paired = filtering.apply_filters(df_paired)

        if df_paired is None:
            continue

        df_paired["id"] = (df["summary"]+df["article"]).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        output_filename = f"/home/ec2-user/data/processed_ccnews_summaries/{datetime.now().strftime('%Y%m%d%Y-%H%M%S')}.parquet"
        df_paired.to_parquet(output_filename)
        logger.info(f"Dumping processed summaries to {output_filename}")


def align_df(df_paired, df):
    df_paired["entity_maintext"] = df.loc[df_paired["article_id"]]["maintext_entity_text"].reset_index(drop=True)
    df_paired["entity_type_lead"] = df.loc[df_paired["summary_id"]]["lead_entity_type"].reset_index(drop=True)
    df_paired["entity_type_maintext"] = df.loc[df_paired["article_id"]]["maintext_entity_type"].reset_index(drop=True)
    df_paired["summary_title"] = df.loc[df_paired["summary_id"]]["title"].reset_index(drop=True)
    df_paired["article_title"] = df.loc[df_paired["article_id"]]["title"].reset_index(drop=True)
    df_paired["summary_domain"] = df.loc[df_paired["summary_id"]]["source_domain"].reset_index(drop=True)
    df_paired["article_domain"] = df.loc[df_paired["article_id"]]["source_domain"].reset_index(drop=True)
    df_paired.loc[:, "summary_maintext"] = df.loc[df_paired["summary_id"]]["maintext"].reset_index(drop=True)
    df_paired.loc[:, "entity_lead"] = df.loc[df_paired["summary_id"]]["lead_entity_text"].reset_index(drop=True)
    return df_paired


def create_summary_article_pairs_from_one_window(df, window_indices, min_cluster_size=2):
    input_articles = df.loc[window_indices]["maintext"].to_list()
    logger.info(f"Encoding {len(input_articles)} articles using sentence transformer")
    window_data = data_util.encode_sentence_transformer(input_articles, multiprocess=False)
    logger.info("Retrieving soft clusters")
    clusters = retrieval.retrieve_soft_clusters(window_data, df.index, min_cluster_size=min_cluster_size)
    logger.info("Generating summary-article pairs")
    summary_article_dataset = text_util.construct_summary_article_pairs(clusters, df)
    return summary_article_dataset


@click.command()
@click.option('--data',
              default='/home/ec2-user/data/cc-news-sliding-windows/20210310.20210314.parquet', help='Input data')
@click.option('--time_window_in_days', default=4, help='The size of time_window_in_days')
@click.option('--min_cluster_size', default=2, help='The min number of articles in a valid cluster.')
def main(data, time_window_in_days, min_cluster_size):
    logger.info('loading data')
    df = pd.read_parquet(data)
    logger.info(f'{len(df)} data loaded')

    df.loc[:, "date_publish"] = pd.to_datetime(df["date_publish"], infer_datetime_format=True)
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
    df.set_index('id', drop=True, inplace=True, verify_integrity=True)

    create_self_supervised_summaries(df, time_window_in_days, min_cluster_size)
    # create_self_supervised_summaries_from_window_parquet(df, min_cluster_size, data)


if __name__ == '__main__':
    main()
