import pandas as pd
import json
import os
from gradio_app import process_clustering
from unittest.mock import MagicMock, patch
import numpy as np

@patch('gradio_app.visualize_clusters_interactive')
@patch('gradio_app.run_clustering_pipeline')
def test_process_clustering(mock_run_pipeline, mock_visualize):
    # Setup mock return values
    dummy_df = pd.DataFrame({'col': [1, 2]})
    dummy_plot = "gradio_output_plot.png"
    dummy_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    # Create dummy plot file
    with open(dummy_plot, 'w') as f:
        f.write("dummy")
        
    mock_run_pipeline.return_value = (dummy_df, dummy_plot, dummy_embeddings)
    mock_visualize.return_value = "dummy_figure_object"

    # Setup dummy inputs
    csv_path = 'data/input.csv'
    tags_path = 'data/tags.json'
    
    # Mock file object for CSV
    mock_csv = MagicMock()
    mock_csv.name = csv_path
    
    # Mock file object for Tags
    mock_tags = MagicMock()
    mock_tags.name = tags_path
    
    # 1. Test with CSV and Tags File
    print("Testing with Tags File...")
    valid_result = process_clustering(mock_csv, None, mock_tags, model_name="dummy_model", normalize=True)
    df, csv_out, plot_out = valid_result
    
    assert isinstance(df, pd.DataFrame)
    assert os.path.exists(csv_out)
    assert plot_out == "dummy_figure_object"
    print("Pass: Tags File")

    # 2. Test with CSV and Tags Text
    print("Testing with Tags Text...")
    with open(tags_path, 'r') as f:
        tags_text = f.read()
        
    valid_result_text = process_clustering(mock_csv, tags_text, None, model_name="dummy_model", normalize=True)
    df2, csv_out2, plot_out2 = valid_result_text
    
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == df.shape
    print("Pass: Tags Text")
    
    # Cleanup
    if os.path.exists("gradio_output.csv"):
        os.remove("gradio_output.csv")
    if os.path.exists("gradio_output_plot.png"):
        os.remove("gradio_output_plot.png")

if __name__ == "__main__":
    test_process_clustering()
    print("All tests passed!")
