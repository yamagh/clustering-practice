import gradio as gr
import pandas as pd
import json
import os
from src.pipeline import run_clustering_pipeline
from src.visualization import visualize_clusters_interactive

def process_clustering(csv_file, tags_input, tags_file):
    # 1. CSV の読み込み
    if csv_file is None:
        raise gr.Error("CSVファイルは必須です。")
    
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        raise gr.Error(f"CSVの読み込みに失敗しました: {str(e)}")

    # 2. タグの読み込み
    tags = []
    # 優先順位: ファイル > テキスト
    if tags_file is not None:
        try:
            with open(tags_file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tags = data.get('tags', [])
        except Exception as e:
            raise gr.Error(f"タグJSONファイルの読み込みに失敗しました: {str(e)}")
    elif tags_input and tags_input.strip():
        try:
            data = json.loads(tags_input)
            tags = data.get('tags', [])
        except Exception as e:
            raise gr.Error(f"タグJSONテキストの解析に失敗しました: {str(e)}")
    
    if not tags:
        raise gr.Error("タグが必要です (JSONテキストまたはファイル)。")

    # 3. パイプラインの実行
    try:
        # プロット用の一時パスを作成
        output_plot_path = "gradio_output_plot.png"
        
        result_df, plot_path, embeddings = run_clustering_pipeline(df, tags, output_plot_path=output_plot_path)
        
        # 結果 CSV をダウンロード用に保存
        output_csv_path = "gradio_output.csv"
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        # インタラクティブプロットの生成
        fig = visualize_clusters_interactive(result_df, embeddings, tags)
        
        return result_df, output_csv_path, fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"クラスタリング中にエラーが発生しました: {str(e)}")


def create_demo():
    with gr.Blocks(title="クラスタリング練習 UI") as demo:
        gr.Markdown("# クラスタリング練習アプリ")
        gr.Markdown("データとタグをアップロードしてクラスタリング分析を実行します。")
        
        with gr.Row():
            with gr.Column():
                csv_input = gr.File(label="入力 CSV のアップロード", file_types=[".csv"])
                
                gr.Markdown("### タグ設定")
                gr.Markdown("JSON ファイルをアップロードするか、JSON コンテンツを貼り付けてください。")
                tags_file = gr.File(label="タグ JSON のアップロード", file_types=[".json"])
                tags_text = gr.Code(label="またはタグ JSON を貼り付け", language="json", lines=5)
                
                submit_btn = gr.Button("分析を実行", variant="primary")
            
            with gr.Column():
                plot_output = gr.Plot(label="クラスター可視化")
                
        with gr.Row():
            result_table = gr.Dataframe(label="結果", interactive=False)
            
        with gr.Row():
            download_btn = gr.File(label="結果 CSV のダウンロード")

        submit_btn.click(
            fn=process_clustering,
            inputs=[csv_input, tags_text, tags_file],
            outputs=[result_table, download_btn, plot_output]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
