import os
import json
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import logging
import torch


# def save_metrics(metrics, output_dir, filename):
#     """
#     Save evaluation metrics to a JSON file.

#     Args:
#         metrics (dict): The evaluation metrics to save.
#         output_dir (str): Directory to save the file.
#         filename (str): Name of the file to save the metrics.
#     """
#     os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
#     file_path = os.path.join(output_dir, filename)
#     with open(file_path, "w") as f:
#         json.dump(metrics, f, indent=4)

def save_metrics(metrics, folder_manager, filename):
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): The metrics to save.
        output_dir (str): Directory where the file will be saved.
        filename (str): Name of the JSON file.
    """

    # Use the metrics directory from the folder manager
    filepath = os.path.join(folder_manager.metrics_dir, filename)

    # Ensure metrics are serializable (convert Torch tensors to lists if necessary)
    serializable_metrics = {
        key: (value.tolist() if isinstance(value, torch.Tensor) else value)
        for key, value in metrics.items()
    }

    # Save the metrics to the JSON file
    os.makedirs(folder_manager.metrics_dir, exist_ok=True)  # Ensure the directory exists
    with open(filepath, "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    logging.info(f"Metrics saved to {filepath}")



def load_image(url):
    """
    Load an image from a URL and return a PIL Image object.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))  # Resize to match ViLT input requirements
        return img
    except Exception as e:
        logging.error(f"Failed to load image from URL {url}: {e}")
        return None

def load_images(df):
    """
    Apply image loading function to a DataFrame.
    """
    df["image"] = df["image_url"].apply(load_image)
    return df[df["image"].notnull()]

def filter_invalid_rows(df, image_column='image', min_valid_rows=10):
    """
    Filters out rows with missing or invalid images.
    Ensures the dataset has at least `min_valid_rows` remaining.
    """
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            if row[image_column] is not None:  # Ensure image exists
                valid_rows.append(True)
            else:
                valid_rows.append(False)
                logging.error(f"Invalid row at index {idx}: 'image'")
        except Exception as e:
            logging.error(f"Error at index {idx}: {e}")
            valid_rows.append(False)

    filtered_df = df[valid_rows]
    if len(filtered_df) < min_valid_rows:
        logging.warning(f"Filtered dataset has less than {min_valid_rows} valid rows.")
    return filtered_df

