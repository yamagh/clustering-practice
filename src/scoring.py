from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


from functools import lru_cache

@lru_cache(maxsize=1)
def get_model(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    return SentenceTransformer(model_name)

def calculate_tag_scores(texts: list[str], tags: list[dict], model=None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    タグごとに複数のキーワードを使用して、テキストとタグの間の意味的類似度スコアを計算します。
    タグのスコアは、定義されたすべてのテキストの中で最大の類似度スコアとなります。

    Args:
        texts: 入力テキストのリスト。
        tags: タグ定義のリスト。各辞書には 'name' と 'texts' が必要です。
              例: [{'name': 'Economy', 'texts': ['Money', 'Finance']}, ...]
        model: 事前ロードされた SentenceTransformer モデル。

    Returns:
        以下のタプル:
        - 各タグの類似度スコアを含む DataFrame (0.0 - 1.0)。
        - テキスト埋め込みの Numpy 配列。
    """
    if model is None:
        model = get_model()

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

    # pandas を使用するかループ処理で効率的に行えます。
    # n_tags が小さいため、ループ処理で十分明確です。
    # text_to_tag_index を numpy に変換してマスキングすることもできますが、単純な反復処理が堅牢です。
    
    # total_n_tag_keywords が大きい場合、集計に pandas アプローチを使用する方がきれいかもしれませんが、
    # 想定される規模を考えると生 numpy がおそらく高速です。
    
    # raw_similarity_matrix の列を反復処理し、対応するタグの最大スコアを更新
    for col_idx, tag_idx in enumerate(text_to_tag_index):
        aggregated_scores[:, tag_idx] = np.maximum(aggregated_scores[:, tag_idx], raw_similarity_matrix[:, col_idx])

    # 5. DataFrame の作成
    tag_names = [t['name'] for t in tags]
    scores_df = pd.DataFrame(aggregated_scores, columns=tag_names)
    
    # 値を 0.0 - 1.0 にクリップ
    scores_df = scores_df.clip(0.0, 1.0)

    return scores_df, text_embeddings
