import faiss
import logging
import tqdm

logger = logging.getLogger(__name__)


def retrieve_soft_clusters(
    window_data, ids, embedding_dim=768, threshold=0.9, min_cluster_size=2
):
    assert window_data.shape[0] == len(ids)
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)  # build the index
    faiss.normalize_L2(window_data)
    faiss_index.add(window_data)
    num_docs = window_data.shape[0]

    logger.info("starting clustering")
    clusters = set()
    batch_size = 5000
    for i in tqdm.trange(num_docs // batch_size):
        distances, indices = faiss_index.search(
            window_data[i * batch_size : (i + 1) * batch_size], 20
        )
        cluster_list = [
            frozenset(ids[ind[d > threshold]]) for ind, d in zip(indices, distances)
        ]
        for c in cluster_list:
            if len(c) >= min_cluster_size:
                clusters.add(c)

    logger.info(f"Found {len(clusters)} clusters.")
    del faiss_index

    return list(clusters)


def retrieve_from_one_doc(
    faiss_index, doc_embedding, embedding_dim=768, threshold=0.85
):
    # search for the top 10 summaries most similar to the doc_embedding
    distances, indices = faiss_index.search(
        doc_embedding.reshape([1, embedding_dim]), 50
    )
    cluster_indices = frozenset(indices[distances > threshold])
    return cluster_indices
