"""Calculates the Mint scores.
"""

from __future__ import print_function
import spacy
from spacy.util import minibatch
from scipy.stats import hmean
from pylcs import lcs


def ngrams_all(tokens, max_n=4):
    """Returns the sets of 1-grams, 2-grams, ..., n-grams as a list of lists.
    >>> ngrams = ngrams_all(['a', 'b', 'c', 'd', 'e'])
    >>> [sorted(list(x)) for x in ngrams]
    [[('a',), ('b',), ('c',), ('d',), ('e',)], [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')], [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e')], [('a', 'b', 'c', 'd'), ('b', 'c', 'd', 'e')]]
    """
    if len(tokens) < max_n:
        tokens += ["__DUMMY__"] * (max_n - len(tokens))
    offsets = [tokens[i:] for i in range(max_n)]
    return [zip(*offsets[:i]) for i in range(1, max_n + 1)]


def ngram_overlaps(x_tok, y_tok, smooth=True):
    """
    >>> ngram_overlaps(['a'], ['a', 'a', 'a', 'b', 'b', 'c'])
    [(2.3333333333333335, 6), (0.7777777777777778, 5), (0.25925925925925924, 4), (0.08641975308641975, 3)]
    >>> ngram_overlaps(['a'], ['a', 'a', 'a', 'b', 'b', 'c'], smooth=False)
    [(3, 6), (0, 5), (0, 4), (0, 3)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'The Supreme Court Thursday reserved its verdict on a batch of pleas which have raised questions'.split())
    [(13.333333333333334, 16), (10.777777777777779, 15), (7.9259259259259265, 14), (5.3086419753086425, 13)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'The Supreme Court Thursday reserved its verdict on a batch of pleas which have raised questions'.split(), smooth=False)
    [(14, 16), (11, 15), (8, 14), (5, 13)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'Supreme Court has reserved a verdict on the batch of the pleas which have raised some questions'.split(),)
    [(8.666666666666666, 17), (3.888888888888889, 16), (1.2962962962962963, 15), (0.43209876543209874, 14)]
    >>> ngram_overlaps('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'Supreme Court has reserved a verdict on the batch of the pleas which have raised some questions'.split(), smooth=False)
    [(11, 17), (3, 16), (0, 15), (0, 14)]
    """
    nx = ngrams_all(x_tok, 5)
    ny = ngrams_all(y_tok, 5)

    def count_match(x, y):
        return sum(1 for ngram in y if ngram in x)

    match_counts = [0]  # zero-gram
    for x, y in zip(nx, ny):
        match_counts.append(count_match(set(x), list(y)))

    # Smoothing, Chen & Cherry (2005), method 5
    if smooth:
        match_counts[0] = match_counts[1] + 1
        for i in range(1, 5):
            match_counts[i] = (
                match_counts[i - 1] + match_counts[i] + match_counts[i + 1]
            ) / 3.0

    lengths = range(len(y_tok), len(y_tok) - 4, -1)
    return list(zip(match_counts[1:5], lengths))


def mint(x_tok, y_tok):
    """
    >>> mint(['a'], ['a', 'a', 'a', 'b', 'b', 'c'])
    {'p1': (2.3333333333333335, 6, 0.3888888888888889), 'p2': (0.7777777777777778, 5, 0.15555555555555556), 'p3': (0.25925925925925924, 4, 0.06481481481481481), 'p4': (0.08641975308641975, 3, 0.028806584362139915), 'lcsr': (1, 6, 0.16666666666666666), 'mint': 0.9232456140350878}
    >>> mint(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
    {'p1': (4.0, 4, 1.0), 'p2': (3.0, 3, 1.0), 'p3': (2.0, 2, 1.0), 'p4': (1.0, 1, 1.0), 'lcsr': (4, 4, 1.0), 'mint': 0.0}
    >>> mint(['x'], ['a', 'b', 'c', 'd'])
    {'p1': (0.3333333333333333, 4, 0.08333333333333333), 'p2': (0.1111111111111111, 3, 0.037037037037037035), 'p3': (0.037037037037037035, 2, 0.018518518518518517), 'p4': (0.012345679012345678, 1, 0.012345679012345678), 'lcsr': (0, 4, 0.0), 'mint': 1.0}
    >>> mint('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'The Supreme Court Thursday reserved its verdict on a batch of pleas which have raised questions'.split())
    {'p1': (13.333333333333334, 16, 0.8333333333333334), 'p2': (10.777777777777779, 15, 0.7185185185185186), 'p3': (7.9259259259259265, 14, 0.5661375661375662), 'p4': (5.3086419753086425, 13, 0.40835707502374174), 'lcsr': (14, 16, 0.875), 'mint': 0.3710535235740673}
    >>> mint('The Supreme Court Thursday reserved its decision on a batch of pleas that have raised questions'.split(), 'Supreme Court has reserved a verdict on the batch of the pleas which have raised some questions'.split())
    {'p1': (8.666666666666666, 17, 0.5098039215686274), 'p2': (3.888888888888889, 16, 0.24305555555555555), 'p3': (1.2962962962962963, 15, 0.08641975308641975), 'p4': (0.43209876543209874, 14, 0.030864197530864196), 'lcsr': (10, 17, 0.5882352941176471), 'mint': 0.9033765130600977}
    """
    overlaps = ngram_overlaps(x_tok, y_tok)
    lcs_ = lcs(x_tok, y_tok)
    lcsr = lcs_ / len(y_tok)
    mint = (
        1.0
        if lcsr == 0.0
        else 1.0 - hmean([num / denom for num, denom in overlaps] + [lcsr])
    )
    d = {
        "p" + str(i + 1): (num, denom, num / denom)
        for i, (num, denom) in enumerate(overlaps)
    }
    d.update({"lcsr": (lcs_, len(y_tok), lcsr), "mint": mint})
    return d


def process_batch(nlp, batch):
    src, tgt = zip(*batch)
    src_docs, tgt_docs = nlp.pipe(src), nlp.pipe(tgt)

    def tokenize(doc):
        return [tok.lower_ for tok in doc]

    s_tokens = [tokenize(s) for s in src_docs]
    t_tokens = [tokenize(t) for t in tgt_docs]

    return [mint(s, t) for s, t in zip(s_tokens, t_tokens)]


def evaluate_mint(summaries, articles, batch_size=32):
    def yield_input(src, tgt):
        for s, t in zip(src, tgt):
            yield s.strip(), t.strip()

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    batches = minibatch(yield_input(articles, summaries), size=batch_size)
    scores = []
    for b in batches:
        m = process_batch(nlp, b)
        scores.extend(m)
    return scores


def evaluate_mint_on_df(df, summary_col, article_col, batch_size=32):
    summaries = df[summary_col].tolist()
    articles = df[article_col].tolist()

    return evaluate_mint(summaries, articles, batch_size)
