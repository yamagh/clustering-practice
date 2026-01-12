import unicodedata
import re

def normalize_text(text: str) -> str:
    """
    日本語テキストの正規化を行います。
    1. Unicode正規化 (NFKC)
    2. 空白文字の削除（前後）
    3. 連続する改行や空白の置換
    
    Args:
        text (str): 入力テキスト
        
    Returns:
        str: 正規化されたテキスト
    """
    if not isinstance(text, str):
        return ""
        
    # Unicode正規化 (NFKC) - 半角カタカナを全角になど
    text = unicodedata.normalize('NFKC', text)
    
    # 前後の空白削除
    text = text.strip()
    
    # 連続する空白を1つに
    text = re.sub(r'\s+', ' ', text)
    
    return text
