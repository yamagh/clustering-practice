import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def assign_primary_cluster(scores_df: pd.DataFrame) -> pd.Series:
    """
    Assign the primary cluster based on the highest tag score.
    Returns a Series with the tag name as the cluster ID.
    """
    return scores_df.idxmax(axis=1)

def generate_secondary_clusters(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    primary_cluster_col: str, 
    n_clusters: int = 3
) -> pd.Series:
    """
    Generate secondary clusters within each primary cluster using K-Means.
    
    Args:
        df: DataFrame containing the primary cluster assignment.
        embeddings: Text embeddings corresponding to the DataFrame rows.
        primary_cluster_col: Name of the column containing the primary cluster IDs.
        n_clusters: Number of clusters to generate for each primary cluster.
        
    Returns:
        Series containing the secondary cluster IDs (integers).
    """
    secondary_clusters = pd.Series(index=df.index, dtype=int)
    
    unique_primary_clusters = df[primary_cluster_col].unique()
    
    for p_cluster in unique_primary_clusters:
        # Filter indices for the current primary cluster
        mask = df[primary_cluster_col] == p_cluster
        indices = df.index[mask]
        
        if len(indices) < n_clusters:
            # If fewer items than clusters, just assign 0...n
            cluster_labels = np.arange(len(indices))
        else:
            # Get corresponding embeddings
            # Assuming embeddings are aligned with df index (0..N)
            # If df index is not reset, we might need to be careful. 
            # Ideally embeddings is a numpy array matching df order.
            current_embeddings = embeddings[mask]
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(current_embeddings)
            
        secondary_clusters.loc[indices] = cluster_labels
        
    return secondary_clusters
