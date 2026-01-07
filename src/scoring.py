from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def get_model(model_name: str = 'all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def calculate_tag_scores(texts: list[str], tags: list[str], model = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate semantic similarity scores between texts and tags.

    Args:
        texts: List of input texts.
        tags: List of tags to score against.
        model: Pre-loaded SentenceTransformer model.

    Returns:
        Tuple containing:
        - DataFrame containing similarity scores for each tag (0.0 - 1.0).
        - Numpy array of text embeddings.
    """
    if model is None:
        model = get_model()

    # Encode texts and tags
    text_embeddings = model.encode(texts)
    tag_embeddings = model.encode(tags)

    # Calculate cosine similarity
    # similarity_matrix shape: (n_texts, n_tags)
    similarity_matrix = cosine_similarity(text_embeddings, tag_embeddings)

    # Create DataFrame
    scores_df = pd.DataFrame(similarity_matrix, columns=tags)
    
    # Clip values to 0.0 - 1.0
    scores_df = scores_df.clip(0.0, 1.0)

    return scores_df, text_embeddings
