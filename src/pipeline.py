import pandas as pd
import json
import os
from src.scoring import calculate_tag_scores
from src.clustering import assign_primary_cluster, generate_secondary_clusters
from src.analysis import calculate_representativeness
from src.visualization import visualize_clusters

import numpy as np

def run_clustering_pipeline(input_df: pd.DataFrame, tags: list[dict], output_plot_path: str = 'data/cluster_visualization.png') -> tuple[pd.DataFrame, str, np.ndarray]:
    """
    Runs the full clustering pipeline:
    1. Calculate Tag Scores
    2. Assign Primary Cluster
    3. Generate Secondary Clusters
    4. Calculate Position Scores
    5. Visualize
    
    Args:
        input_df: Input DataFrame containing the text to analyze.
        tags: List of tag dictionaries.
        output_plot_path: Path to save the visualization image.
        
    Returns:
        A tuple containing:
        - The processed DataFrame with scores and cluster assignments.
        - The path to the saved visualization image.
    """
    
    print("Step 1: Calculating Tag Relevance Scores...")
    # Calculate scores and get embeddings
    # Assuming '文章' is the text column.
    if '文章' not in input_df.columns:
        raise ValueError("Input DataFrame must contain a '文章' column.")
        
    texts = input_df['文章'].fillna('').tolist()
    scores_df, embeddings = calculate_tag_scores(texts, tags)
    
    # Merge scores into main DF
    # Reset index to ensure alignment if input_df has non-standard index
    df_combined = pd.concat([input_df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    print("Step 2: Assigning Primary Cluster...")
    tag_names = [t['name'] for t in tags]
    primary_clusters = assign_primary_cluster(df_combined[tag_names])
    df_combined['第1クラスター番号'] = primary_clusters

    print("Step 3: Generating Secondary Clusters...")
    # Generating secondary clusters
    # Note: n_clusters=2 is hardcoded in main.py, keeping it same here or we can make it an arg.
    secondary_clusters = generate_secondary_clusters(
        df_combined, 
        embeddings, 
        primary_cluster_col='第1クラスター番号',
        n_clusters=2 
    )
    df_combined['第2クラスター番号'] = secondary_clusters

    print("Step 4: Calculating Position Scores...")
    df_combined = calculate_representativeness(
        df_combined,
        embeddings,
        primary_cluster_col='第1クラスター番号',
        secondary_cluster_col='第2クラスター番号'
    )
    
    # Rename columns to match requirements
    df_combined = df_combined.rename(columns={'position_score': '位置スコア'})
    
    # Reorder columns
    # Keep original columns usually, but ensure the required ones are present
    # Logic from main.py:
    cols = ['ID', '文章'] + tag_names + ['第1クラスター番号', '第2クラスター番号', '位置スコア']
    
    # Validate if ID exists, otherwise just keep what we have or add it?
    # main.py assumes ID exists.
    available_cols = [c for c in cols if c in df_combined.columns]
    # If there are other columns in input, we might want to keep them or strictly follow the output format.
    # main.py does: df_final = df_combined[cols]
    
    df_final = df_combined[available_cols]

    print("Step 5: Visualizing Clusters...")
    visualize_clusters(df_final, embeddings, tags, output_path=output_plot_path)
    
    return df_final, output_plot_path, embeddings
