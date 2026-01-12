import pandas as pd
import json
import os
from src.scoring import calculate_tag_scores
from src.clustering import assign_primary_cluster, generate_secondary_clusters
from src.analysis import calculate_representativeness

from src.visualization import visualize_clusters

def load_tags(path: str = 'tags.json') -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['tags']

def main():
    # パス設定
    input_path = 'data/input.csv'
    output_path = 'data/output.csv'
    tags_path = 'tags.json'
    vis_output_path = 'data/cluster_visualization.png'

    print("データを読み込んでいます...")
    if not os.path.exists(input_path):
        print(f"エラー: {input_path} が見つかりません。")
        return
        
    df = pd.read_csv(input_path)
    tags = load_tags(tags_path)

    # パイプラインの実行
    from src.pipeline import run_clustering_pipeline
    df_final, _, _ = run_clustering_pipeline(df, tags, output_plot_path=vis_output_path)

    print(f"結果を {output_path} に保存しています...")
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig') # Excel互換性のための utf-8-sig
    print("完了！")

if __name__ == "__main__":
    main()
