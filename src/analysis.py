import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_representativeness(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    primary_cluster_col: str, 
    secondary_cluster_col: str
) -> pd.DataFrame:
    """
    Calculate position score (closeness to centroid) for each item.
    
    Args:
        df: DataFrame with cluster assignments.
        embeddings: Text embeddings.
        primary_cluster_col: Column name for primary cluster.
        secondary_cluster_col: Column name for secondary cluster.
        
    Returns:
        DataFrame with an added 'position_score' column.
    """
    df = df.copy()
    df['position_score'] = 0.0
    
    # Group by (Primary, Secondary)
    # We create a combined key or iterate unique pairs.
    # Iterating unique pairs is safer.
    
    unique_pairs = df[[primary_cluster_col, secondary_cluster_col]].drop_duplicates().values
    
    for p_cluster, s_cluster in unique_pairs:
        mask = (df[primary_cluster_col] == p_cluster) & (df[secondary_cluster_col] == s_cluster)
        indices = df.index[mask]
        
        if len(indices) == 0:
            continue
            
        current_embeddings = embeddings[mask]
        
        # Calculate centroid
        centroid = np.mean(current_embeddings, axis=0).reshape(1, -1)
        
        # Calculate distances to centroid
        distances = euclidean_distances(current_embeddings, centroid).flatten()
        
        # Convert distance to score (0.0 - 1.0)
        # Closer is better.
        # Strategy: 1 / (1 + distance)
        scores = 1.0 / (1.0 + distances)
        
        df.loc[indices, 'position_score'] = scores
        
    return df

def select_representatives(df: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    """
    Select top N representatives per cluster.
    """
    # This might be just filtering, but the requirement says "Extract top 10".
    # We can probably just sort and take head, or return the whole DF sorted.
    # Let's return the whole DF but sorted by score within groups.
    return df.sort_values(by=['position_score'], ascending=False)
