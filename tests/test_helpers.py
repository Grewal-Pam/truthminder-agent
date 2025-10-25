# tests/test_helpers.py
import os
import json
from utils.helpers import save_metrics


def test_save_metrics():
    # Define dummy metrics
    metrics = {"accuracy": 0.9, "precision": 0.85, "recall": 0.87, "f1_score": 0.86}
    output_dir = "test_results"
    filename = "test_metrics.json"

    # Call the save_metrics function
    save_metrics(metrics, output_dir, filename)

    # Verify the file exists
    file_path = os.path.join(output_dir, filename)
    assert os.path.exists(file_path)

    # Verify file content
    with open(file_path, "r") as f:
        saved_metrics = json.load(f)
    assert saved_metrics == metrics

    # Clean up test artifacts
    os.remove(file_path)
    os.rmdir(output_dir)
