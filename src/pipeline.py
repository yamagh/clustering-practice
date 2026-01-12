import pandas as pd
import json
import os
from src.scoring import calculate_tag_scores
from src.clustering import assign_primary_cluster, generate_secondary_clusters
from src.analysis import calculate_representativeness
from src.analysis import calculate_representativeness
from src.visualization import visualize_clusters
from src.preprocessing import normalize_text

import numpy as np

def run_clustering_pipeline(input_df: pd.DataFrame, tags: list[dict], output_plot_path: str = 'data/cluster_visualization.png') -> tuple[pd.DataFrame, str, np.ndarray]:
    """
    クラスタリングパイプライン全体を実行します:
    1. タグスコアの計算
    2. 第1クラスターの割り当て (Primary Cluster)
    3. 第2クラスターの生成 (Secondary Clusters)
    4. 位置スコア (Position Scores) の計算
    5. 可視化
    
    Args:
        input_df: 分析対象のテキストを含む入力 DataFrame。
        tags: タグ定義の辞書リスト。
        output_plot_path: 可視化画像の保存先パス。
        
    Returns:
        以下のタプル:
        - スコアとクラスター割り当てを含む処理済み DataFrame。
        - 保存された可視化画像のパス。
        - テキストの埋め込みベクトル (embeddings)。
    """
    
    print("ステップ 1: タグ関連性スコアを計算中...")
    # スコアの計算と埋め込みベクトルの取得
    # '文章' がテキストカラムであると仮定
    if '文章' not in input_df.columns:
        raise ValueError("入力 DataFrame には '文章' カラムが必要です。")
        
    # 前処理の適用
    raw_texts = input_df['文章'].fillna('').tolist()
    texts = [normalize_text(t) for t in raw_texts]
    
    scores_df, embeddings = calculate_tag_scores(texts, tags)
    
    # スコアをメインの DF にマージ
    # input_df のインデックスが標準でない場合に備えてリセット
    df_combined = pd.concat([input_df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    print("ステップ 2: 第1クラスターを割り当て中...")
    tag_names = [t['name'] for t in tags]
    primary_clusters = assign_primary_cluster(df_combined[tag_names])
    df_combined['第1クラスター番号'] = primary_clusters

    print("ステップ 3: 第2クラスターを生成中...")
    # 第2クラスターの生成
    # 注: n_clusters=2 は main.py でハードコードされているため、ここでも同じにしていますが、引数化も可能です。
    secondary_clusters = generate_secondary_clusters(
        df_combined, 
        embeddings, 
        primary_cluster_col='第1クラスター番号',
        n_clusters=2
    )
    df_combined['第2クラスター番号'] = secondary_clusters

    print("ステップ 4: 位置スコアを計算中...")
    df_combined = calculate_representativeness(
        df_combined,
        embeddings,
        primary_cluster_col='第1クラスター番号',
        secondary_cluster_col='第2クラスター番号'
    )
    
    # 要件に合わせてカラム名を変更
    df_combined = df_combined.rename(columns={'position_score': '位置スコア'})
    
    # カラムの並べ替え
    # 通常は元のカラムを維持しますが、必須カラムが存在することを確認します
    # main.py のロジック:
    cols = ['ID', '文章'] + tag_names + ['第1クラスター番号', '第2クラスター番号', '位置スコア']
    
    # IDが存在するか確認し、なければあるものだけを使用
    # main.py は ID が存在することを前提としています。
    available_cols = [c for c in cols if c in df_combined.columns]
    # 入力に他のカラムがある場合、それらを保持するか、出力フォーマットに厳密に従うか
    # main.py では: df_final = df_combined[cols]
    
    df_final = df_combined[available_cols]

    print("ステップ 5: クラスターを可視化中...")
    visualize_clusters(df_final, embeddings, tags, output_path=output_plot_path)
    
    return df_final, output_plot_path, embeddings
