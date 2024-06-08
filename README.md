## CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization (NAACL 2024)
This Repo contains the code for the paper "CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization."

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
pip install news-please
pip install lxml=4.8.0

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

