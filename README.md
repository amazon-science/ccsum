## CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization (NAACL 2024)
This Repo contains the code for the paper ["CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization."](https://aclanthology.org/2024.naacl-long.406/)

CCSum is a large-scale and high-quality dataset for abstractive news summarization.
It contains 1.3 million pairs of articles and reference summaries derived from 35 million news articles from CommonCrawl News.
In creating this dataset, we cluster CommonCrawl News articles into news events from which we generate candidate article-summary pairs and apply strict filtering and a Bayesian optimization method that eliminates 99% of the candidate summaries.
The human evaluation shows the proposed dataset has higher quality---in terms of factual consistency, informativeness, and coherence---than established abstractive summarization datasets.

## CCSum Dataset
We release the summary, article url and meta-data of the CCSum dataset. The articles needs to be downloaded from CC-News.

### Summary and meta-data
The summaries and meta-data can be found in Huggingface [`ccsum/ccsum_summary_only`](https://huggingface.co/datasets/ccsum/ccsum_summary_only).

```python
from datasets import load_dataset
# Load the full dataset (both abstractive and extractive)
dataset = load_dataset("ccsum/ccsum_summary_only")

# abstractive subset of the dataset
dataset_abstractive = dataset.filter(lambda x: x["abstractiveness_bin"] == "high")

# extractive subset of the dataset
dataset_extractive = dataset.filter(lambda x: x["abstractiveness_bin"] == "low")
```

### Download News Articles from CCNews
Create python environment:
```
# create conda environment
conda create -y -n ccsum python=3.8 && conda activate ccsum

# Install newsplease crawler
pip install news-please==1.5.48
pip install lxml==4.8.0

pip install tqdm datasets

# Download news articles from cc-news
# The script only download urls present in the ccsum dataset
cd ./data
python commoncrawl.py
```
The downloaded articles will be stored at `./data/ccnews_downloaded_articles/`.

Then, run the following code to align the downloaded articles with the summaries:
```
cd ./data
python align_article_with_summary.py

```
The final dataset will be stored in `./data/ccsum_aligned`

Please reach out to us if you encounter any issues with downloading the dataset.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License. See the LICENSE file.

## Cite: Bibtex
@inproceedings{jiang-dreyer-2024-ccsum,
    title = "{CCS}um: A Large-Scale and High-Quality Dataset for Abstractive News Summarization",
    author = "Jiang, Xiang  and
      Dreyer, Markus",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.406",
    pages = "7306--7336",
    abstract = "Training a supervised news summarization model requires large amounts of high-quality training data consisting of news articles paired with reference summaries. However, obtaining such data is costly, and existing datasets contain considerable amount of noise. We present a new large-scale and high-quality dataset for supervised abstractive news summarization containing 1.3 million training samples, which we call CCSum. In creating this dataset, we take advantage of the journalistic inverted-pyramid style in news writing: In some articles, the first sentence can be considered a summary of the reported story. Accordingly, among 35 million CommonCrawl News articles, we identify pairs of articles about the same news story and use one article{'}s first sentence as the summary for the other article. To ensure high quality, we apply strict filters whose parameters we optimize using Bayesian optimization. We show that the resulting dataset is more factual and informative than established summarization datasets; less than 1{\%} of the summaries have major factual inconsistencies with the corresponding news articles, compared to 5.5{\%} to 15.4{\%} in existing datasets, according to our human evaluation. Summarization models trained on our dataset are more favored compared to those trained on CNN/Daily Mail. The proposed dataset can open new opportunities for future research in abstractive summarization.",
}

