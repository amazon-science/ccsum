## CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization (NAACL 2024)
This Repo contains the code for the paper "CCSum: A Large-Scale and High-Quality Dataset for Abstractive News Summarization."

CCSum is a large-scale and high-quality dataset for abstractive news summarization.
It contains 1.3 million pairs of articles and reference summaries derived from 35 million news articles from CommonCrawl News.
In creating this dataset, we cluster CommonCrawl News articles into news events from which we generate candidate article-summary pairs and apply strict filtering and a Bayesian optimization method that eliminates 99% of the candidate summaries.
The human evaluation shows the proposed dataset has higher quality---in terms of factual consistency, informativeness, and coherence---than established abstractive summarization datasets.

## CCSum Dataset
The dataset can be found in Huggingface [`ccsum/CCSum`](https://huggingface.co/datasets/ccsum/CCSum).

```python
from datasets import load_dataset
# Load the full dataset (both abstractive and extractive)
dataset = load_dataset("ccsum/CCSum")

# abstractive subset of the dataset
dataset_abstractive = dataset.filter(lambda x: x["abstractiveness_bin"] == "high")

# extractive subset of the dataset
dataset_extractive = dataset.filter(lambda x: x["abstractiveness_bin"] == "low")
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License. See the LICENSE file.

