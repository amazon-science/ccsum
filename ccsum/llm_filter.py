import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
from ccsum import (
    llm_models,
)
import logging
import pandas as pd
import click
import torch
import hashlib
from datetime import datetime
import gc
import time

logger = logging.getLogger(__name__)


def evaluate_factuality(df, input_filename):
    output_filename = f"/home/ec2-user/data/processed_ccnews_summaries_llm/{input_filename.split('/')[-1]}"
    model, tokenizer = llm_models.load_model(model_name="google/flan-t5-xl")

    # Evaluate the answers
    summary_eval_prompts = [
        f"Article: {article[:1000]}\nBased on the paragraph above can we conclude that: {summary}"
        for article, summary in zip(df["article"].tolist(), df["summary"].tolist())
    ]
    summary_factuality = llm_models.batch_query(
        model,
        tokenizer,
        summary_eval_prompts,
        decoding_args={
            "max_new_tokens": 5,
            "do_sample": False,
            "num_return_sequences": 1,
        },
        batch_size=16,
    )
    df["summary_factuality_flant5xl"] = summary_factuality

    # grammatical
    grammatical_prompts = [
        f"Would the following sentence, by the strictest standards, be considered correct by a linguist?\n{summary}\nOptions: acceptable or unacceptable\nAnswer:"
        for summary in df["summary"].tolist()
    ]
    grammatical_flant5xl = llm_models.batch_query(
        model,
        tokenizer,
        grammatical_prompts,
        decoding_args={
            "max_new_tokens": 5,
            "do_sample": False,
            "num_return_sequences": 1,
        },
        batch_size=32,
    )
    df["grammatical_flant5xl"] = grammatical_flant5xl

    logger.info(f"Dumping processed summaries to {output_filename}")
    df.to_parquet(output_filename)


@click.command()
@click.option(
    "--data",
    default="/home/ec2-user/data/processed_ccnews_summaries/2018-01-01 0:0:0-2018-07-01 0:0:0/",
    help="Input data",
)
def main(data):
    logger.info("loading data")
    df = pd.read_parquet(data)
    logger.info("data loaded")

    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    logger.info(f"{len(df)} summaries loaded.")
    evaluate_factuality(df, data)


if __name__ == "__main__":
    main()
