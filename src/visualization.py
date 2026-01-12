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
    PCA 2D 射影を使用してクラスタリング結果とタグを可視化します。
    
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
    
    # 2. PCA 次元削減
    # 共通空間に合わせるためにテキスト埋め込みとタグ埋め込みを結合
    # (オプション: テキストのみで適合させてタグがどこに来るか見る、または両方で適合させる。
    # 両方で適合させると、両方がうまく表現されます。)
    # テキストの方が数が多いので、主にテキストで適合させ、タグを射影します。
    
    pca = PCA(n_components=2)
    # テキストで適合
    text_coords = pca.fit_transform(text_embeddings)
    # タグを変換
    tag_coords = pca.transform(tag_vectors)
    
    # 3. プロット
    plt.figure(figsize=(12, 10))
    
    # 色付けに '第2クラスター番号' を使用
    # NaN または -1 がある場合の処理
    clusters = df['第2クラスター番号'].fillna(-1)
    unique_clusters = sorted(clusters.unique())
    
    # カラーマップの生成
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        plt.scatter(
            text_coords[mask, 0], 
            text_coords[mask, 1], 
            label=f'クラスター {cluster_id}',
            alpha=0.6,
            s=30
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
        
    plt.title('クラスタリング可視化 (2D PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
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
    
    # 2. PCA
    pca = PCA(n_components=2)
    text_coords = pca.fit_transform(text_embeddings)
    tag_coords = pca.transform(tag_vectors)
    
    # 3. Plotly Figure の作成
    fig = go.Figure()
    
    # テキストの追加
    # クラスターごとに離散的な色を使用できます
    clusters = df['第2クラスター番号'].fillna(-1).astype(str)
    
    # Plotly Express などで容易にするためにプロット用 DataFrame を作成するか、
    # graph_objects でトレースを制御するためにループ処理するか。
    unique_clusters = sorted(clusters.unique(), key=lambda x: int(float(x)) if x.replace('.','',1).isdigit() or x.lstrip('-').isdigit() else 999)
    
    # 色の定義
    # Plotly のデフォルトシーケンスを使用
    colors = px.colors.qualitative.Plotly
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        
        # ホバーテキストの準備
        # ホバーテキストにはテキストのスニペットを含めるべき
        hover_texts = df.loc[mask, '文章'].apply(lambda x: x[:100] + "..." if len(str(x)) > 100 else str(x)).tolist()
        
        fig.add_trace(go.Scatter(
            x=text_coords[mask, 0],
            y=text_coords[mask, 1],
            mode='markers',
            name=f'クラスター {cluster_id}',
            marker=dict(
                size=8,
                opacity=0.7
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
        title="クラスタリング可視化 (インタラクティブ)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="凡例",
        template="plotly_white",
        autosize=True
    )
    
    return fig
