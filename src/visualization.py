import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.scoring import get_model # Reuse the model loader if needed, or just pass embeddings

def visualize_clusters(df: pd.DataFrame, text_embeddings: np.ndarray, tags: list[dict], output_path: str = 'data/cluster_visualization.png'):
    """
    Visualize clustering results and tags using PCA 2D projection.
    
    Args:
        df: DataFrame containing '第2クラスター番号' and other data.
        text_embeddings: Numpy array of sentence embeddings for the texts.
        tags: List of tag definitions to calculate tag positions.
        output_path: Path to save the resulting image.
    """
    print("Generating visualization...")
    
    # 1. Calculate Tag Embeddings
    # We need the model to encode tags. 
    # To avoid re-loading model excessively, ideally it should be passed in, 
    # but for simplicity we'll load it here or assume lightweight.
    # Actually, we can reuse the one from scoring if we refactor, but strict separation is okay too.
    model = get_model()
    
    tag_names = []
    tag_vectors = []
    
    for tag in tags:
        # Average the embeddings of all keywords for the tag
        keywords = tag['texts']
        # encode returns (n_keywords, 384)
        kw_embeddings = model.encode(keywords) 
        # Average vector
        tag_mean_embedding = np.mean(kw_embeddings, axis=0)
        
        tag_names.append(tag['name'])
        tag_vectors.append(tag_mean_embedding)
        
    tag_vectors = np.array(tag_vectors)
    
    # 2. PCA Dimensionality Reduction
    # Combine text embeddings and tag embeddings to fit the common space
    # (Optional: Fit only on texts to see where tags fall, or fit on both. 
    # Fitting on both ensures both are well-represented.)
    # Let's fit on texts primarily, as there are more of them, and project tags.
    
    pca = PCA(n_components=2)
    # Fit on texts
    text_coords = pca.fit_transform(text_embeddings)
    # Transform tags
    tag_coords = pca.transform(tag_vectors)
    
    # 3. Plotting
    plt.figure(figsize=(12, 10))
    
    # Use '第2クラスター番号' for coloring
    # Handle NaN or -1 if any
    clusters = df['第2クラスター番号'].fillna(-1)
    unique_clusters = sorted(clusters.unique())
    
    # Generate a colormap
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        plt.scatter(
            text_coords[mask, 0], 
            text_coords[mask, 1], 
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=30
        )
        
    # Plot Tags
    plt.scatter(
        tag_coords[:, 0], 
        tag_coords[:, 1], 
        c='red', 
        marker='X', 
        s=200, 
        label='Tags', 
        edgecolors='black'
    )
    
    # Annotate Tags
    for i, name in enumerate(tag_names):
        plt.text(
            tag_coords[i, 0], 
            tag_coords[i, 1], 
            name, 
            fontsize=12, 
            fontweight='bold', 
            color='darkred',
            ha='right', 
            va='bottom'
        )
        
    plt.title('Clustering Visualization (2D PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")
