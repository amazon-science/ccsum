import spacy
import itertools
from tqdm import tqdm

import logging
from ccsum import data, entity_util
import uuid

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
nlp.add_pipe("sentencizer")

existing_pair = set()


def construct_summary_article_pairs(clusters, df):
    summary_article_dataset = data.SummaryArticlePairDataset()

    for cluster in tqdm(clusters, desc="Constructing paired data"):
        summary, article, summary_id, article_id = (
            construct_summary_article_pairs_within_cluster(list(cluster), df)
        )

        if len(summary) == 0:
            continue

        summary_article_dataset.add_summary_article_pairs(
            summary, article, summary_id, article_id, [uuid.uuid4().hex] * len(summary)
        )

    return summary_article_dataset


def construct_summary_article_pairs_within_cluster(article_ids, df):
    comb = itertools.permutations(article_ids, 2)
    summary_ids, article_ids = zip(*comb)
    summary_ids = list(summary_ids)
    article_ids = list(article_ids)

    filtered_summary_ids, filtered_article_ids = [], []

    skip_cnt = 0

    global existing_pair
    if len(existing_pair) > 1000000000:
        existing_pair = set()

    for s, a in zip(summary_ids, article_ids):
        if (s, a) not in existing_pair:
            filtered_summary_ids.append(s)
            filtered_article_ids.append(a)
            existing_pair.add((s, a))
        else:
            skip_cnt += 1

    # if skip_cnt != 0:
    # logger.info(f"Skipped {skip_cnt} existing candidate summaries.")
    summaries = df.loc[filtered_summary_ids]["lead_sentence_cleaned"].to_list()
    articles = df.loc[filtered_article_ids]["maintext"].to_list()
    return summaries, articles, filtered_summary_ids, filtered_article_ids


def extract_lead_sentence(article):
    docs = list(nlp.pipe([article]))
    sentences = list(docs[0].sents)
    if len(sentences) > 0:
        return sentences[0].text
    else:
        return ""


def evaluate_entity_precision(articles, abstracts):
    ents1 = entity_util.get_entities(nlp, articles)
    ents2 = entity_util.get_entities(nlp, abstracts)

    def precision(e1, e2):
        unmatched = e2 - e1
        Z = len(e2) or 1e-24
        p = 1.0 - float(len(unmatched)) / Z
        return p

    entity_precision = [precision(e1, e2) for (e1, e2) in zip(ents1, ents2)]
    return entity_precision
