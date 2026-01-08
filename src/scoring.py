from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def get_model(model_name: str = 'all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def calculate_tag_scores(texts: list[str], tags: list[dict], model=None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate semantic similarity scores between texts and tags using multiple keywords per tag.
    The score for a tag is the maximum similarity score among all its defined texts.

    Args:
        texts: List of input texts.
        tags: List of tag definitions. Each dict should have 'name' and 'texts'.
              Example: [{'name': 'Economy', 'texts': ['Money', 'Finance']}, ...]
        model: Pre-loaded SentenceTransformer model.

    Returns:
        Tuple containing:
        - DataFrame containing similarity scores for each tag (0.0 - 1.0).
        - Numpy array of text embeddings.
    """
    if model is None:
        model = get_model()

    # Encode input texts
    text_embeddings = model.encode(texts)

    # 1. Collect all tag texts and map them back to tag names
    all_tag_texts = []
    text_to_tag_index = [] # Maps index of text in all_tag_texts to index of tag in tags list

    for i, tag_def in enumerate(tags):
        tag_texts = tag_def['texts']
        all_tag_texts.extend(tag_texts)
        text_to_tag_index.extend([i] * len(tag_texts))

    # 2. Encode all tag texts
    tag_embeddings = model.encode(all_tag_texts)

    # 3. Calculate cosine similarity (texts vs all tag keywords)
    # Shape: (n_texts, total_n_tag_keywords)
    raw_similarity_matrix = cosine_similarity(text_embeddings, tag_embeddings)

    # 4. Aggregate scores for each tag (take max over the tag's keywords)
    n_texts = len(texts)
    n_tags = len(tags)
    aggregated_scores = np.zeros((n_texts, n_tags))

    # We can do this efficiently using pandas or looping. 
    # Since n_tags is small, looping is fine and clear.
    # Convert text_to_tag_index to numpy for masking if needed, but simple iteration is robust.
    
    # Using a pandas approach for aggregation might be cleaner if total_n_tag_keywords is large,
    # but raw numpy is likely faster given the likely scale.
    
    # Iterate over columns of raw_similarity_matrix and update max score for the corresponding tag
    for col_idx, tag_idx in enumerate(text_to_tag_index):
        aggregated_scores[:, tag_idx] = np.maximum(aggregated_scores[:, tag_idx], raw_similarity_matrix[:, col_idx])

    # 5. Create DataFrame
    tag_names = [t['name'] for t in tags]
    scores_df = pd.DataFrame(aggregated_scores, columns=tag_names)
    
    # Clip values to 0.0 - 1.0
    scores_df = scores_df.clip(0.0, 1.0)

    return scores_df, text_embeddings
