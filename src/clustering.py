import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def assign_primary_cluster(scores_df: pd.DataFrame) -> pd.Series:
    """
    最も高いタグスコアに基づいて第1クラスターを割り当てます。
    タグ名をクラスターIDとして持つ Series を返します。
    """
    return scores_df.idxmax(axis=1)

def generate_secondary_clusters(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    primary_cluster_col: str, 
    n_clusters: int = 3
) -> pd.Series:
    """
    K-Means を使用して、各第1クラスター内に第2クラスターを生成します。
    
    Args:
        df: 第1クラスター割り当てを含む DataFrame。
        embeddings: DataFrame の行に対応するテキスト埋め込み。
        primary_cluster_col: 第1クラスターIDを含むカラム名。
        n_clusters: 各第1クラスターに対して生成するクラスター数。
        
    Returns:
        第2クラスターID (整数) を含む Series。
    """
    secondary_clusters = pd.Series(index=df.index, dtype=int)
    
    unique_primary_clusters = df[primary_cluster_col].unique()
    
    for p_cluster in unique_primary_clusters:
        # 現在の第1クラスターのインデックスをフィルタリング
        mask = df[primary_cluster_col] == p_cluster
        indices = df.index[mask]
        
        if len(indices) < n_clusters:
            # クラスター数よりアイテム数が少ない場合、単純に 0...n を割り当てる
            cluster_labels = np.arange(len(indices))
        else:
            # 対応する埋め込みを取得
            # 埋め込みが df のインデックス (0..N) と一致していると仮定
            # df のインデックスがリセットされていない場合、注意が必要。
            # 理想的には埋め込みは df の順序と一致する numpy 配列であるべき。
            current_embeddings = embeddings[mask]
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(current_embeddings)
            
        secondary_clusters.loc[indices] = cluster_labels
        
    return secondary_clusters
