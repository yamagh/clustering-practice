from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


from functools import lru_cache

@lru_cache(maxsize=1)
@lru_cache(maxsize=5)
def get_model(model_name: str = 'cl-nagoya/ruri-v3-70m'):
    """
    指定された名前の SentenceTransformer モデルをロードして返します。
    LRUキャッシュを使用して、最近使用されたモデルをメモリに保持します。
    """
    print(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)

def calculate_tag_scores(texts: list[str], tags: list[dict], model=None, model_name: str = 'cl-nagoya/ruri-v3-70m', normalize: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """
    タグごとに複数のキーワードを使用して、テキストとタグの間の意味的類似度スコアを計算します。
    タグのスコアは、定義されたすべてのテキストの中で最大の類似度スコアとなります。

    Args:
        texts: 入力テキストのリスト。
        tags: タグ定義のリスト。各辞書には 'name' と 'texts' が必要です。
              例: [{'name': 'Economy', 'texts': ['Money', 'Finance']}, ...]
        model: 事前ロードされた SentenceTransformer モデル。Noneの場合は内部でロードします。
        model_name: ロードするモデルの名前。modelがNoneの場合に使用されます。
        normalize: コサイン類似度をMin-Max正規化するかどうか。Trueの場合、行ごとに0-1にスケールされます。

    Returns:
        以下のタプル:
        - 各タグの類似度スコアを含む DataFrame (0.0 - 1.0)。
        - テキスト埋め込みの Numpy 配列。
    """
    if model is None:
        model = get_model(model_name=model_name)

    # 入力テキストのエンコード
    text_embeddings = model.encode(texts)

    # 1. すべてのタグテキストを収集し、タグ名にマッピング
    all_tag_texts = []
    text_to_tag_index = [] # all_tag_texts 内のテキストのインデックスを tags リストのタグのインデックスにマッピング

    for i, tag_def in enumerate(tags):
        tag_texts = tag_def['texts']
        all_tag_texts.extend(tag_texts)
        text_to_tag_index.extend([i] * len(tag_texts))

    # 2. すべてのタグテキストをエンコード
    tag_embeddings = model.encode(all_tag_texts)

    # 3. コサイン類似度の計算 (テキスト vs 全タグキーワード)
    # 形状: (n_texts, total_n_tag_keywords)
    raw_similarity_matrix = cosine_similarity(text_embeddings, tag_embeddings)

    # 4. 各タグのスコアを集計 (タグのキーワードの最大値を取得)
    n_texts = len(texts)
    n_tags = len(tags)
    aggregated_scores = np.zeros((n_texts, n_tags))

    # タグごとに処理
    for t_idx in range(n_tags):
        # このタグに対応する列のインデックスを取得
        # text_to_tag_index はリストなので、numpy array にして検索
        indices = np.where(np.array(text_to_tag_index) == t_idx)[0]
        
        if len(indices) == 0:
            continue
            
        # (n_texts, n_tag_keywords)
        sub_matrix = raw_similarity_matrix[:, indices]
        
        # 行ごとに Top-K を計算
        # キーワード数が3未満なら全平均、それ以上なら上位3つの平均
        k = 3
        if sub_matrix.shape[1] <= k:
            tag_scores = np.mean(sub_matrix, axis=1)
        else:
            # sort は昇順なので、後ろから k 個取る
            tag_scores = np.mean(np.sort(sub_matrix, axis=1)[:, -k:], axis=1)
            
        aggregated_scores[:, t_idx] = tag_scores

    if normalize:
        # 5. 正規化処理 (Min-Max Normalization per row)
        # 各文章について、スコアの分布を 0.0 - 1.0 に広げることで、
        # タグ間の相対的な重要度を明確にする。
        # raw_score の差がわずかでも、min=0, max=1 になるようにスケーリングする。
        
        # 行ごとの最小値と最大値を計算 (n_texts, 1)
        row_mins = aggregated_scores.min(axis=1, keepdims=True)
        row_maxs = aggregated_scores.max(axis=1, keepdims=True)
        
        # ゼロ除算を防ぐための小さなイプシロン
        epsilon = 1e-10
        
        # Min-Max Normalization: (x - min) / (max - min)
        # 差が非常に小さい（max ≈ min）場合は 0除算になる可能性があるため対策
        ranges = row_maxs - row_mins
        # 範囲がほぼ0の場合は、すべて0にするか、元のスコアを維持するかだが、
        # ここでは一律に少し小さな値にするか、0にする。
        # しかし、コサイン類似度で完全に同一になることは稀。
        
        # ブロードキャストで計算
        final_scores = (aggregated_scores - row_mins) / (ranges + epsilon)
    else:
        # 正規化しない場合はそのまま使用
        final_scores = aggregated_scores
    
    # 6. DataFrame の作成
    tag_names = [t['name'] for t in tags]
    scores_df = pd.DataFrame(final_scores, columns=tag_names)
    
    # 値を 0.0 - 1.0 にクリップ (計算誤差対策)
    scores_df = scores_df.clip(0.0, 1.0)

    return scores_df, text_embeddings
