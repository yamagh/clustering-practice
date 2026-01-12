import gradio as gr
import pandas as pd
import json
import os
from src.pipeline import run_clustering_pipeline
from src.visualization import visualize_clusters_interactive

def process_clustering(csv_file, tags_input, tags_file):
    # 1. Load CSV
    if csv_file is None:
        raise gr.Error("CSV File is required.")
    
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        raise gr.Error(f"Failed to read CSV: {str(e)}")

    # 2. Load Tags
    tags = []
    # Priority: File > Text
    if tags_file is not None:
        try:
            with open(tags_file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tags = data.get('tags', [])
        except Exception as e:
            raise gr.Error(f"Failed to read Tags JSON file: {str(e)}")
    elif tags_input and tags_input.strip():
        try:
            data = json.loads(tags_input)
            tags = data.get('tags', [])
        except Exception as e:
            raise gr.Error(f"Failed to parse Tags JSON text: {str(e)}")
    
    if not tags:
        raise gr.Error("Tags are required (either JSON text or file).")

    # 3. Run Pipeline
    try:
        # Create a temporary path for the plot
        output_plot_path = "gradio_output_plot.png"
        
        result_df, plot_path, embeddings = run_clustering_pipeline(df, tags, output_plot_path=output_plot_path)
        
        # Save Result CSV for download
        output_csv_path = "gradio_output.csv"
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        # Generate Interactive Plot
        fig = visualize_clusters_interactive(result_df, embeddings, tags)
        
        return result_df, output_csv_path, fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error during clustering: {str(e)}")


def create_demo():
    with gr.Blocks(title="Clustering Practice UI") as demo:
        gr.Markdown("# Clustering Practice App")
        gr.Markdown("Upload your data and tags to run the clustering analysis.")
        
        with gr.Row():
            with gr.Column():
                csv_input = gr.File(label="Upload Input CSV", file_types=[".csv"])
                
                gr.Markdown("### Tags Configuration")
                gr.Markdown("Upload a JSON file OR paste JSON content.")
                tags_file = gr.File(label="Upload Tags JSON", file_types=[".json"])
                tags_text = gr.Code(label="Or Paste Tags JSON", language="json", lines=5)
                
                submit_btn = gr.Button("Run Analysis", variant="primary")
            
            with gr.Column():
                plot_output = gr.Plot(label="Cluster Visualization")
                
        with gr.Row():
            result_table = gr.Dataframe(label="Results", interactive=False)
            
        with gr.Row():
            download_btn = gr.File(label="Download Output CSV")

        submit_btn.click(
            fn=process_clustering,
            inputs=[csv_input, tags_text, tags_file],
            outputs=[result_table, download_btn, plot_output]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
