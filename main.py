import pandas as pd
import json
import os
from src.scoring import calculate_tag_scores
from src.clustering import assign_primary_cluster, generate_secondary_clusters
from src.analysis import calculate_representativeness

def load_tags(path: str = 'tags.json') -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['tags']

def main():
    # Paths
    input_path = 'data/input.csv'
    output_path = 'data/output.csv'
    tags_path = 'tags.json'

    print("Loading data...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    tags = load_tags(tags_path)

    print("Step 1: Calculating Tag Relevance Scores...")
    # Calculate scores and get embeddings
    # Assuming '文章' is the text column.
    texts = df['文章'].fillna('').tolist()
    scores_df, embeddings = calculate_tag_scores(texts, tags)
    
    # Merge scores into main DF
    df_combined = pd.concat([df, scores_df], axis=1)

    print("Step 2: Assigning Primary Cluster...")
    # Pass only the score columns
    tag_names = [t['name'] for t in tags]
    primary_clusters = assign_primary_cluster(df_combined[tag_names])
    df_combined['第1クラスター番号'] = primary_clusters

    print("Step 3: Generating Secondary Clusters...")
    secondary_clusters = generate_secondary_clusters(
        df_combined, 
        embeddings, 
        primary_cluster_col='第1クラスター番号',
        n_clusters=2 # Default to 2 for small dummy data, maybe make configurable
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
    # ID, 文章, Tag Scores..., Primary Cluster, Secondary Cluster, Position Score
    # Required Output columns:
    # ID
    # Tag Scores (Already there)
    # 第1クラスター番号
    # 第2クラスター番号
    # 位置スコア -> 'position_score' -> Rename to '位置スコア'
    
    df_combined = df_combined.rename(columns={'position_score': '位置スコア'})
    
    # Reorder columns
    # Reorder columns
    cols = ['ID', '文章'] + tag_names + ['第1クラスター番号', '第2クラスター番号', '位置スコア']
    # Check if '文章' is needed in output? The requirement table doesn't explicitly list '文章' but says "All calculated indicators combined".
    # Usually input columns are preserved. "出力CSVには、計算されたすべての指標を結合します" (Combine all calculated indicators to output CSV).
    # The table in user request lists: ID, Tag Scores, 1st Cluster, 2nd Cluster, Position Score.
    # It doesn't explicitly list Text. But it's usually good to keep it.
    # I'll include '文章' for clarity unless strictly forbidden.
    
    df_final = df_combined[cols]

    print(f"Saving output to {output_path}...")
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
    print("Done!")

if __name__ == "__main__":
    main()
