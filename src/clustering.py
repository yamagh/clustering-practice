import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
        n_samples = len(indices)
        
        if n_samples < 3:
            # サンプル数が少なすぎる場合はクラスター化しない（全て0にするか、個別に割り当てる）
            # ここでは単純に 0 を割り当てます
            cluster_labels = np.zeros(n_samples, dtype=int)
        else:
            current_embeddings = embeddings[mask]
            
            # 最適なクラスター数を探索 (シルエット分析)
            # 探索範囲: 2 から min(5, サンプル数 - 1)
            # silhouette_score は n_labels < n_samples である必要があるため、max_k は n_samples - 1
            max_k = min(5, n_samples - 1)
            best_k = 2
            best_score = -1
            best_labels = None
            
            # デフォルト (k=2) の計算
            # n_samples が 2 の場合、k=2 は n_labels == n_samples となり silhouette_score は計算不可だが、
            # ここではループに入らないため計算されない。
            # ただし、k=2 の結果自体は有効な場合がある（単にスコア計算ができないだけ）。
            if n_samples >= 2:
                 kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                 best_labels = kmeans.fit_predict(current_embeddings)
            else:
                 # n_samples=1 の場合など (前の条件で除外されているはずだが念のため)
                 best_labels = np.zeros(n_samples, dtype=int)
            
            if max_k > 2:
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(current_embeddings)
                    
                    # シルエットスコアの計算
                    # 注: 本来は距離行列を事前に計算したほうが高速ですが、データ量が少ないため都度計算でも可
                    try:
                        score = silhouette_score(current_embeddings, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                            best_labels = labels
                    except ValueError:
                        # n_labels < 2 or n_labels >= n_samples の場合など
                        continue
            
            cluster_labels = best_labels
            
            cluster_labels = best_labels
            
        secondary_clusters.loc[indices] = cluster_labels
        
    return secondary_clusters
