import pandas as pd
import json
import os
from gradio_app import process_clustering
from unittest.mock import MagicMock

def test_process_clustering():
    # Setup dummy inputs
    csv_path = 'data/input.csv'
    tags_path = 'tags.json'
    
    # Mock file object for CSV
    mock_csv = MagicMock()
    mock_csv.name = csv_path
    
    # Mock file object for Tags
    mock_tags = MagicMock()
    mock_tags.name = tags_path
    
    # 1. Test with CSV and Tags File
    print("Testing with Tags File...")
    valid_result = process_clustering(mock_csv, None, mock_tags)
    df, csv_out, plot_out = valid_result
    
    assert isinstance(df, pd.DataFrame)
    assert os.path.exists(csv_out)
    assert os.path.exists(plot_out)
    print("Pass: Tags File")

    # 2. Test with CSV and Tags Text
    print("Testing with Tags Text...")
    with open(tags_path, 'r') as f:
        tags_text = f.read()
        
    valid_result_text = process_clustering(mock_csv, tags_text, None)
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
