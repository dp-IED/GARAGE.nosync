import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def compute_window_similarity(
    window_embeddings: Dict[int, Dict[str, Any]], k: int = 5
) -> List[Tuple[int, int, float, float]]:
    if len(window_embeddings) == 0:
        return []

    window_indices = sorted(window_embeddings.keys())
    embeddings_list = [window_embeddings[idx]["embedding"] for idx in window_indices]
    embeddings_array = np.array(embeddings_list)

    N = len(window_indices)
    if N < 2:
        return []

    similarity_edges = []

    for i, window_i in enumerate(window_indices):
        embedding_i = embeddings_array[i : i + 1]
        similarities = cosine_similarity(embedding_i, embeddings_array)[0]
        distances = cdist(embedding_i, embeddings_array, metric="euclidean")[0]
        similarities[i] = -np.inf
        top_k_indices = np.argsort(similarities)[::-1][:k]

        for j in top_k_indices:
            if j != i:
                window_j = window_indices[j]
                cosine_sim = float(similarities[j])
                euclidean_dist = float(distances[j])
                similarity_edges.append(
                    (window_i, window_j, cosine_sim, euclidean_dist)
                )

    return similarity_edges
