from ccsum import data_util
import pandas as pd
import spacy
import sklearn

nlp = spacy.load("en_core_web_sm")


def get_summary_article_similarity_at_sentence_level(summary: str, article: str):
    sentences = [s.text for s in nlp(article).sents]
    sentence_embedding = data_util.encode_sentence_transformer(
        [s for s in sentences], multiprocess=False
    )
    summary_embedding = data_util.encode_sentence_transformer(
        [summary], multiprocess=False
    )
    similarity = sklearn.metrics.pairwise.cosine_similarity(
        summary_embedding, sentence_embedding
    ).flatten()
    return similarity, sentences


def get_similarity(t1: list, t2: list):
    e1 = data_util.encode_sentence_transformer(t1, multiprocess=False)
    e2 = data_util.encode_sentence_transformer(t2, multiprocess=False)
    similarity = 1 - sklearn.metrics.pairwise.paired_cosine_distances(e1, e2)
    return similarity
