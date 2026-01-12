import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from src.scoring import get_model # Reuse the model loader if needed, or just pass embeddings
import plotly.graph_objects as go
import plotly.express as px

def visualize_clusters(df: pd.DataFrame, text_embeddings: np.ndarray, tags: list[dict], output_path: str = 'data/cluster_visualization.png'):
    """
    UMAP 2D 射影を使用してクラスタリング結果とタグを可視化します。
    
    Args:
        df: '第2クラスター番号' などのデータを含む DataFrame。
        text_embeddings: テキストの文埋め込みベクトル (Numpy配列)。
        tags: タグ位置を計算するためのタグ定義リスト。
        output_path: 結果画像を保存するパス。
    """
    print("可視化を生成中...")
    
    # 1. タグ埋め込みの計算
    # タグをエンコードするためにモデルが必要です。
    # 過度な再ロードを避けるため、理想的には渡されるべきですが、
    # 簡単のためここでロードするか、軽量であると仮定します。
    # 実際には、リファクタリングすれば scoring から再利用できますが、厳密な分離も許容範囲です。
    model = get_model()
    
    tag_names = []
    tag_vectors = []
    
    for tag in tags:
        # タグの全キーワードの埋め込みを平均化
        keywords = tag['texts']
        # encode は (n_keywords, 384) を返す
        kw_embeddings = model.encode(keywords) 
        # 平均ベクトル
        tag_mean_embedding = np.mean(kw_embeddings, axis=0)
        
        tag_names.append(tag['name'])
        tag_vectors.append(tag_mean_embedding)
        
    tag_vectors = np.array(tag_vectors)
    
    # 2. UMAP 次元削減
    # 共通空間に合わせるためにテキスト埋め込みとタグ埋め込みを結合
    import umap
    n_samples = len(text_embeddings)
    n_neighbors = min(15, n_samples - 1) if n_samples > 1 else 1
    if n_neighbors < 2: n_neighbors = 2
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    
    # テキストで適合
    text_coords = reducer.fit_transform(text_embeddings)
    # タグを変換
    tag_coords = reducer.transform(tag_vectors)
    
    # 3. プロット
    plt.figure(figsize=(12, 10))
    
    # 色付けとラベル生成
    # 第1クラスター番号がある場合は、それと組み合わせてラベルを作成
    if '第1クラスター番号' in df.columns:
        # 欠損値処理
        p_clusters = df['第1クラスター番号'].fillna('Unknown')
        s_clusters = df['第2クラスター番号'].fillna(-1)
        
        # 組み合わせラベルを作成 (例: "経済 - 1")
        # 第2クラスターが -1 (ノイズ) の場合の扱いも考慮できますが、
        # ここでは単純に文字列結合します
        combined_labels = p_clusters.astype(str) + " - " + s_clusters.astype(str)
        unique_labels = sorted(combined_labels.unique())
        
        clusters = combined_labels
        unique_clusters = unique_labels
    else:
        # 従来通り
        clusters = df['第2クラスター番号'].fillna(-1)
        unique_clusters = sorted(clusters.unique())
    
    # カラーマップの生成 (タブローカラーなどを使用)
    # クラスター数が多い場合は tab20 などに変更することも検討
    if len(unique_clusters) <= 10:
        cmap = plt.cm.get_cmap('tab10', len(unique_clusters))
    else:
        cmap = plt.cm.get_cmap('tab20', len(unique_clusters))
        
    for i, cluster_label in enumerate(unique_clusters):
        mask = clusters == cluster_label
        plt.scatter(
            text_coords[mask, 0], 
            text_coords[mask, 1], 
            label=f'{cluster_label}',
            alpha=0.6,
            s=30,
            color=cmap(i)
        )
        
    # タグのプロット
    plt.scatter(
        tag_coords[:, 0], 
        tag_coords[:, 1], 
        c='red', 
        marker='X', 
        s=200, 
        label='タグ', 
        edgecolors='black'
    )
    
    # タグの注釈
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
        
    plt.title('クラスタリング可視化 (2D UMAP)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可視化を {output_path} に保存しました")

def visualize_clusters_interactive(df: pd.DataFrame, text_embeddings: np.ndarray, tags: list[dict]) -> go.Figure:
    """
    Plotly を使用してクラスターとタグのインタラクティブな 2D 散布図を生成します。

    Args:
        df: '第2クラスター番号', '文章' などを含む DataFrame。
        text_embeddings: テキストの文埋め込みベクトル (Numpy配列)。
        tags: タグ定義リスト。

    Returns:
        go.Figure: Plotly の figure オブジェクト。
    """
    # 1. タグ埋め込みの計算 (静的バージョンと同じロジック)
    model = get_model()
    tag_names = []
    tag_vectors = []
    
    for tag in tags:
        keywords = tag['texts']
        kw_embeddings = model.encode(keywords) 
        tag_mean_embedding = np.mean(kw_embeddings, axis=0)
        tag_names.append(tag['name'])
        tag_vectors.append(tag_mean_embedding)
        
    tag_vectors = np.array(tag_vectors)
    
    # 2. UMAP
    import umap
    n_samples = len(text_embeddings)
    n_neighbors = min(15, n_samples - 1) if n_samples > 1 else 1
    if n_neighbors < 2: n_neighbors = 2
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    
    text_coords = reducer.fit_transform(text_embeddings)
    tag_coords = reducer.transform(tag_vectors)
    
    # 3. Plotly Figure の作成
    fig = go.Figure()
    
    # テキストの追加
    # テキストの追加
    # クラスターラベルの生成
    if '第1クラスター番号' in df.columns:
        p_clusters = df['第1クラスター番号'].fillna('Unknown').astype(str)
        s_clusters = df['第2クラスター番号'].fillna(-1).astype(str)
        combined_labels = p_clusters + " - " + s_clusters
        clusters = combined_labels
    else:
        clusters = df['第2クラスター番号'].fillna(-1).astype(str)
    
    unique_clusters = sorted(clusters.unique())
    
    # 色の定義
    # Plotly のデフォルトシーケンスを使用
    colors = px.colors.qualitative.Plotly
    
    for i, cluster_label in enumerate(unique_clusters):
        mask = clusters == cluster_label
        
        # ホバーテキストの準備
        # ホバーテキストにはテキストのスニペットを含めるべき
        hover_texts = df.loc[mask, '文章'].apply(lambda x: x[:100] + "..." if len(str(x)) > 100 else str(x)).tolist()
        
        fig.add_trace(go.Scatter(
            x=text_coords[mask, 0],
            y=text_coords[mask, 1],
            mode='markers',
            name=f'{cluster_label}',
            marker=dict(
                size=8,
                opacity=0.7,
                color=colors[i % len(colors)] # 色を割り当て
            ),
            text=hover_texts,
            hoverinfo='text+name'
        ))
        
    # タグの追加
    fig.add_trace(go.Scatter(
        x=tag_coords[:, 0],
        y=tag_coords[:, 1],
        mode='markers+text',
        name='タグ',
        marker=dict(
            symbol='x',
            size=12,
            color='red',
            line=dict(width=2, color='DarkRed')
        ),
        text=tag_names,
        textposition="top center",
        textfont=dict(
            family="sans serif",
            size=14,
            color="DarkRed"
        ),
        hoverinfo='name+text'
    ))
    
    fig.update_layout(
        title="クラスタリング可視化 (インタラクティブ UMAP)",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        legend_title="凡例",
        template="plotly_white",
        autosize=True
    )
    
    return fig
