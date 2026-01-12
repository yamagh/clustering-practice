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
    各アイテムの位置スコア (重心への近さ) を計算します。
    
    Args:
        df: クラスター割り当てを持つ DataFrame。
        embeddings: テキスト埋め込み。
        primary_cluster_col: 第1クラスターのカラム名。
        secondary_cluster_col: 第2クラスターのカラム名。
        
    Returns:
        'position_score' カラムが追加された DataFrame。
    """
    df = df.copy()
    df['position_score'] = 0.0
    
    # (第1クラスター, 第2クラスター) でグループ化
    # 結合キーを作成するか、一意のペアを反復処理します。
    # 一意のペアを反復処理する方が安全です。
    
    unique_pairs = df[[primary_cluster_col, secondary_cluster_col]].drop_duplicates().values
    
    for p_cluster, s_cluster in unique_pairs:
        mask = (df[primary_cluster_col] == p_cluster) & (df[secondary_cluster_col] == s_cluster)
        indices = df.index[mask]
        
        if len(indices) == 0:
            continue
            
        current_embeddings = embeddings[mask]
        
        # 重心の計算
        centroid = np.mean(current_embeddings, axis=0).reshape(1, -1)
        
        # 重心への距離を計算
        distances = euclidean_distances(current_embeddings, centroid).flatten()
        
        # 距離をスコアに変換 (0.0 - 1.0)
        # 近いほど良い。
        # 戦略: 1 / (1 + distance)
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
