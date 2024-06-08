import os
import json
from tqdm import tqdm
from datasets import load_dataset


def load_json(filename):
    if os.stat(filename).st_size == 0:
        return None
    with open(filename, "r", encoding="utf8", errors="ignore") as json_file:
        return json.load(json_file)


def load_json_from_filelist(filelist, show_tqdm=True):

    data = []
    if show_tqdm:
        filelist = tqdm(filelist)
    for f in filelist:
        s = load_json(f)
        if s:
            data.append(s)

    return data


def load_json_from_dir(dir, limitN=None, show_tqdm=True):
    file_list = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(dir))
        for f in fn
        if ".json" in f
    ]
    if limitN:
        file_list = file_list[:limitN]
    return load_json_from_filelist(file_list, show_tqdm)


def main():
    downloaded_articles = load_json_from_dir('./ccnews_download_articles')
    dataset = load_dataset("ccsum/ccsum_summary_only")
    total = len(dataset)
    url_to_article_dict = {a['url']:a['maintext'] for a in downloaded_articles}

    dataset = dataset.map(lambda d: {"article": url_to_article_dict.get(d['url'], '')})
    aligned_dataset = dataset.filter(lambda d: d['article'] != '')
    for split in ['train', 'validation', 'test']:
        print(f"Recovered {len(aligned_dataset[split])} from {len(dataset[split])} articles in ccsum[{split}].")
    aligned_dataset.save_to_disk("ccsum_aligned")


if __name__ == '__main__':
    main()