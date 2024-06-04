from ccsum import regex_util, entity_util, similarity_util, mint, factuality
from simhash import Simhash
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)


def apply_filters(df_paired):
    filters = [
        ending_punctuation,
        summary_article_different_domains,
        summary_word_count,
        summary_at_least_one_entity,
        summary_not_in_article,
        entity_precision_hard,
        entity_precision_soft,
        simhash_filter,
        quotation_filter,
        title_title_sim,
        summary_title_similarity,
        bert_scores_bert_filter,
        bert_scores_bart_filter,
        mint_filter,
    ]
    logger.info(f"Before filter: {len(df_paired)}")
    for f in filters:
        logger.info(f"Applying filter: {f.__name__}")
        df_paired = f(df_paired)

        if len(df_paired) == 0:
            return None
        logger.info(f"Filter completed: {len(df_paired)} summaries remaining")
    return df_paired


def ending_punctuation(df_paired):
    df_paired = df_paired.loc[
        (
            df_paired["summary"].str.endswith(".")
            | df_paired["summary"].str.endswith('"')
            | df_paired["summary"].str.endswith("!")
        )
    ]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def summary_article_different_domains(df_paired):
    df_paired = df_paired.loc[
        df_paired["summary_domain"] != df_paired["article_domain"]
    ]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def summary_word_count(df_paired):
    df_paired.loc[:, "summary_word_count"] = df_paired["summary"].apply(
        lambda x: len(x.split(" ")) + 1
    )
    df_paired = df_paired.reset_index(drop=True)
    df_paired = df_paired.loc[df_paired["summary_word_count"] >= 25]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def summary_at_least_one_entity(df_paired):
    df_paired["summary_entity_count"] = df_paired.entity_lead.apply(lambda x: len(x))
    df_paired = df_paired.loc[df_paired["summary_entity_count"] > 0]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def summary_not_in_article(df_paired):
    df_paired = df_paired.loc[~df_paired.summary.isin(df_paired.article)]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def entity_precision_hard(df_paired):
    df_paired["entity_precision_constraint"] = df_paired.apply(
        lambda x: entity_util.evaluate_entity_precision_constraint(
            x.entity_lead, x.entity_type_lead, x.entity_maintext
        ),
        axis=1,
    )
    df_paired = df_paired.loc[df_paired["entity_precision_constraint"] >= 1]
    df_paired = df_paired.reset_index(drop=True)

    return df_paired


def entity_precision_soft(df_paired):
    df_paired["entity_precision"] = df_paired.apply(
        lambda x: entity_util.evaluate_entity_precision(
            x.entity_lead, x.entity_maintext
        ),
        axis=1,
    )
    df_paired = df_paired.loc[df_paired["entity_precision"] >= 0.89]
    df_paired = df_paired.reset_index(drop=True)

    return df_paired


def simhash_distance(summary_maintext, article):
    a = Simhash(summary_maintext)
    b = Simhash(article)
    return a.distance(b)


def simhash_filter(df_paired):
    with mp.Pool(mp.cpu_count() - 5) as pool:
        df_paired["simhash_distance"] = pool.starmap(
            simhash_distance, zip(df_paired["summary_maintext"], df_paired["article"])
        )

    df_paired = df_paired.loc[df_paired["simhash_distance"] > 10]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def quotation_filter(df_paired):
    with mp.Pool(mp.cpu_count() - 5) as pool:
        df_paired["quotation_precision"] = pool.starmap(
            regex_util.evaluate_quote_precision,
            zip(df_paired["summary"], df_paired["article"]),
        )
    df_paired = df_paired.loc[df_paired["quotation_precision"] >= 1]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def title_title_sim(df_paired):
    df_paired["title-title-similarity"] = similarity_util.get_similarity(
        df_paired["summary_title"].tolist(), df_paired["article_title"].tolist()
    )
    df_paired = df_paired.loc[
        df_paired["title-title-similarity"] >= 0.36148940105061916
    ]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def summary_title_similarity(df_paired):
    df_paired["summary-title-similarity"] = similarity_util.get_similarity(
        df_paired["summary"].tolist(), df_paired["article_title"].tolist()
    )
    df_paired = df_paired.loc[
        df_paired["summary-title-similarity"] >= 0.37502007900162976
    ]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired


def bert_scores_bert_filter(df_paired):
    bert_scores = factuality.evaluate_bert_score(
        df_paired["summary"].tolist(),
        df_paired["article"].tolist(),
        device="cuda",
        model_type="bert-large-uncased",
        batch_size=156,
    )

    for s in bert_scores:
        df_paired[s] = bert_scores[s]

    df_paired = df_paired.loc[df_paired["BERTScore-P (bert-large-uncased)"] >= 0.71]
    df_paired = df_paired.reset_index(drop=True)
    df_paired = df_paired.loc[df_paired["BERTScore-R (bert-large-uncased)"] >= 0.344]
    df_paired = df_paired.reset_index(drop=True)

    return df_paired


def bert_scores_bart_filter(df_paired):
    bert_scores = factuality.evaluate_bert_score(
        df_paired["summary"].tolist(),
        df_paired["article"].tolist(),
        model_type="facebook/bart-large",
        device="cuda",
        batch_size=64,
    )
    for s in bert_scores:
        df_paired[s] = bert_scores[s]

    df_paired = df_paired.loc[df_paired["BERTScore-P (facebook/bart-large)"] >= 0.75]
    df_paired = df_paired.reset_index(drop=True)

    df_paired = df_paired.loc[df_paired["BERTScore-R (facebook/bart-large)"] >= 0.31]
    df_paired = df_paired.reset_index(drop=True)

    return df_paired


def mint_filter(df_paired):
    scores = mint.evaluate_mint_on_df(
        df_paired, summary_col="summary", article_col="article", batch_size=32
    )
    df_paired["mint"] = [s["mint"] for s in scores]
    df_paired["lcsr"] = [s["lcsr"][-1] for s in scores]
    df_paired = df_paired.loc[df_paired["lcsr"] < 0.9]
    df_paired = df_paired.reset_index(drop=True)
    df_paired = df_paired.loc[df_paired["mint"] > 0.2]
    df_paired = df_paired.reset_index(drop=True)
    return df_paired
