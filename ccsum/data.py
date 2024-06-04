from pseudo_ref_research import factuality
import pandas as pd
import numpy as np
import random
import json
import tqdm
import os
import pandas as pd
import re


class SummaryArticlePairDataset:
    def __init__(self):
        self.summaries = []
        self.articles = []
        self.summary_ids = []
        self.article_ids = []
        self.cluster_ids = []
        self.summary_article_set = set()
        self.bert_scores = None

    def add_summary_article_pairs(
        self, summaries, articles, summary_ids, article_ids, cluster_ids
    ):
        for i in range(len(summaries)):
            self.add_summary_article_pair(
                summaries[i],
                articles[i],
                summary_ids[i],
                article_ids[i],
                cluster_ids[i],
            )

        self.validate_data()

    def add_summary_article_pair(
        self, summary, article, summary_id, article_id, cluster_id
    ):
        # Ensure summary is not in article
        if summary in article:
            return

        # Ensure summary-article pairs are unique
        if (summary, article) not in self.summary_article_set:
            self.summary_article_set.add((summary, article))

            self.summaries.append(summary)
            self.articles.append(article)
            self.summary_ids.append(summary_id)
            self.article_ids.append(article_id)
            self.cluster_ids.append(cluster_id)

    def validate_data(self):
        assert len(self.summaries) == len(self.articles)
        assert len(self.summaries) == len(self.summary_ids)
        assert len(self.summaries) == len(self.article_ids)
        assert len(self.summaries) == len(self.cluster_ids)

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return (
            self.summaries[idx],
            self.articles[idx],
            self.summary_ids[idx],
            self.article_ids[idx],
        )

    def evaluate_bert_score(self):
        self.bert_scores = factuality.evaluate_bert_score(self.summaries, self.articles)

    def shuffle(self, seed=0):
        assert self.bert_scores is None
        zipped = list(
            zip(
                self.summaries,
                self.articles,
                self.cluster_ids,
                self.summary_ids,
                self.article_ids,
            )
        )

        random.Random(seed).shuffle(zipped)

        (
            self.summaries,
            self.articles,
            self.cluster_ids,
            self.summary_ids,
            self.article_ids,
        ) = zip(*zipped)

    def take_n(self, n=10000):
        self.summaries = self.summaries[:n]
        self.articles = self.articles[:n]
        self.cluster_ids = self.cluster_ids[:n]
        self.summary_ids = self.summary_ids[:n]
        self.article_ids = self.article_ids[:n]
        if self.bert_scores:
            self.bert_scores = self.bert_scores[:n]

    def to_df(self):
        data_dict = {
            "summary": self.summaries,
            "article": self.articles,
            "cluster_id": self.cluster_ids,
            "summary_id": self.summary_ids,
            "article_id": self.article_ids,
        }

        if self.bert_scores is not None:
            data_dict.update(self.bert_scores)

        df = pd.DataFrame(data_dict)

        return df


def load_jsonl(filename):
    with open(filename, "r", encoding="utf8", errors="ignore") as json_file:
        json_list = list(json_file)
        json_list = [json.loads(j) for j in json_list]
    return json_list


def load_json(filename):
    if os.stat(filename).st_size == 0:
        return None
    with open(filename, "r", encoding="utf8", errors="ignore") as json_file:
        return json.load(json_file)


def load_json_from_filelist(filelist, fn=load_json, show_tqdm=True):

    def load_summary(f):
        try:
            f_json = fn(f)
        except ValueError as e:
            f_json = None
        return f_json

    summaries = []
    if show_tqdm:
        filelist = tqdm(filelist)
    for f in filelist:
        s = load_summary(f)
        if s:
            summaries.append(s)

    return summaries


def load_json_from_dir(dir, limitN=None, show_tqdm=True):
    file_list = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(dir))
        for f in fn
        if ".json" in f
    ]
    if limitN:
        file_list = file_list[:limitN]
    return load_json_from_filelist(file_list, load_json, show_tqdm)


def load_jsonl_from_dir(dir):
    file_list = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(dir))
        for f in fn
        if ".json" in f
    ]
    return load_json_from_filelist(file_list, load_jsonl)


def csv_to_jsonl(csv_filename, jsonl_filename, csv_encoding=None, dtype=None):
    df = pd.read_csv(csv_filename, engine="python", encoding=csv_encoding, dtype=dtype)
    f = open(jsonl_filename, "w")
    print(df.to_json(orient="records", lines=True), file=f, flush=False)
    f.close()


def list_to_jsonl(data_list, jsonl_filename, format="lines"):
    if format == "list":
        f = open(jsonl_filename, "w")
        print(json.dumps(data_list), file=f, flush=False)
        f.close()
    elif format == "lines":
        with open(jsonl_filename, "w") as f:
            for d in data_list:
                json.dump(d, f)
                f.write("\n")
    else:
        raise ValueError(f"Unsupported format {format}")


def dict_to_json(data, filename):
    with open(filename, "w") as fp:
        json.dump(data, fp)


def get_content_between_quotes(text):
    # find all substrings that are between double quotes
    matches = re.findall(r'"([^"]*)"', text)
    return matches


def get_content_between_outermost_quotes(text):
    # find the substring that starts with the first double quote and ends with the last double quote
    match = re.search(r'"(.*)"', text, re.DOTALL)
    if match:
        return match.group(1)  # return the first (and only) group


def escape_quote(text):
    return text.replace('"', '\\"')
