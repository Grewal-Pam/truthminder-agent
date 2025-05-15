import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO
import config
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import FlavaProcessor, FlavaModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
import random
import pickle
import optuna
from PIL import Image
import requests
from io import BytesIO
import pytesseract

from utils.logger import setup_logger
logger = setup_logger("flava_dataloader", log_dir="runs/logs", sampled=False)

class FLAVADataset(Dataset):
    def __init__(self, dataframe, processor, label_type='2_way_label', include_metadata=True, metadata_columns=None):
        self.dataframe = dataframe
        self.processor = processor
        self.label_type = label_type
        self.include_metadata = include_metadata
        self.metadata_columns = metadata_columns if metadata_columns else []

        for col in self.metadata_columns:
            self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['clean_title']
        image = row.get('image_url', None)
        #image = row['image_url']
        if image is None or pd.isna(image):
            logger.error(f"Missing image URL in row {idx}: {row}")
            #raise ValueError(f"Missing image URL in row {idx}")
            return None  # Skip this row
        
        label = torch.tensor(row[self.label_type], dtype=torch.long)
        metadata_values = (
            row[self.metadata_columns].values.astype(float)
            if self.include_metadata else []
        )

        # Fetch and preprocess the image
        image = preprocess_image(image)
        if image is None:
            logger.error(f"Could not process image at URL: {image} in row {idx}")
            #raise ValueError(f"Could not process image at URL: {image} in row {idx}")
            return None
        # Ensure the image tensor is properly shaped
        image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        # Ensure pixel_values is 4D (batch_size, channels, height, width)
        if image_tensor.dim() == 5:
            image_tensor = image_tensor.squeeze(0)
            logger.info(f"Image tensor shape after processing: {image_tensor.shape}")
        inputs = self.processor(
            text=[text],
            images=image,  # Image tensor already processed
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40
        )
        inputs["row_index"] = torch.tensor(idx)  # Track original row index

        # Ensure pixel_values is 4D
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)  # Remove unnecessary batch dimension if present
        inputs["labels"] = label
        if self.include_metadata:
            inputs["metadata"] = torch.tensor(metadata_values, dtype=torch.float32)

        # Debug log
        logger.info(f"Dataset Sample - input_ids: {inputs['input_ids'].shape}, pixel_values: {inputs['pixel_values'].shape}, labels: {inputs['labels'].shape}")
        return inputs


def get_flava_dataloader(df, processor, label_type, batch_size, include_metadata=True, metadata_columns=None):
    logger.info(f"Preparing data loader with label column: {label_type}")
    #print(f"Sample labels: {df[label_type].unique()}")  # Debug label values
    logger.info(f"DataFrame sample for {label_type}:\n{df.head()}")
    logger.info(f"Label counts:\n{df[label_type].value_counts()}")

    dataset = FLAVADataset(df, processor, label_type, include_metadata, metadata_columns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)

def collate_fn_deletelaterifbelowcollateworks(batch):
    # Filter out None values before processing
    batch = [item for item in batch if item is not None and item['labels'].numel() > 0]  
    row_indices = torch.stack([item["row_index"] for item in batch])

    
    if len(batch) == 0:
        return None  # Return None if the entire batch is empty
    
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    pixel_values = torch.cat(
        [item['pixel_values'].unsqueeze(0) if item['pixel_values'].dim() == 3 else item['pixel_values'] for item in batch],
        dim=0
    )
    labels = torch.cat([item['labels'].unsqueeze(0) if item['labels'].dim() == 0 else item['labels'] for item in batch], dim=0)

    metadata = (
        torch.stack([item['metadata'] for item in batch], dim=0)
        if 'metadata' in batch[0] else torch.tensor([], dtype=torch.float32)
    )
    
    logger.info(f"Collated batch, input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels.squeeze(),
        'metadata': metadata if metadata.nelement() > 0 else None,
        'row_index': row_indices
    }


def collate_fn(batch):
    # Filter out invalid items
    batch = [item for item in batch if item is not None and item['labels'].numel() > 0]
    if len(batch) == 0:
        return None  # Skip empty batches

    row_indices = torch.stack([item["row_index"] for item in batch])
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    pixel_values = torch.cat(
        [item['pixel_values'].unsqueeze(0) if item['pixel_values'].dim() == 3 else item['pixel_values'] for item in batch],
        dim=0
    )
    labels = torch.cat([item['labels'].unsqueeze(0) if item['labels'].dim() == 0 else item['labels'] for item in batch], dim=0)

    metadata = (
        torch.stack([item['metadata'] for item in batch], dim=0)
        if 'metadata' in batch[0] else torch.tensor([], dtype=torch.float32)
    )

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'metadata': metadata if metadata.nelement() > 0 else None,
        'row_index': row_indices
    }


def is_placeholder_image(img):
    """
    Detect if an image is likely a placeholder based on simple heuristics.
    This function checks:
      - If the image is completely black or white.
      - If the image has very low variance (almost a uniform color).
      - Optionally, if OCR detects known placeholder phrases.
    """
    try:
        # Convert image to grayscale
        grayscale = img.convert("L")
        # Get pixel extrema (min and max)
        extrema = grayscale.getextrema()
        if extrema == (0, 0):
            logger.warning("Image is completely black.")
            return True
        if extrema == (255, 255):
            logger.warning("Image is completely white.")
            return True

        # Check for low variance: if a single value dominates the histogram.
        histogram = grayscale.histogram()
        total_pixels = sum(histogram)
        max_pixel_count = max(histogram)
        if total_pixels > 0 and (max_pixel_count / total_pixels) > 0.98:
            logger.warning("Image has very low variance; likely a placeholder.")
            return True

        # Optional: OCR-based detection
        try:
            text = pytesseract.image_to_string(img)
            placeholder_keywords = [
                "not found", "unavailable", "error", "placeholder", "no longer available"
            ]
            for keyword in placeholder_keywords:
                if keyword.lower() in text.lower():
                    logger.warning(f"Detected placeholder keyword '{keyword}' in image.")
                    return True
        except Exception as e:
            logger.warning(f"OCR processing failed: {e}")

    except Exception as e:
        logger.error(f"Error during placeholder detection: {e}")
        # If something goes wrong, better treat it as a placeholder to skip it.
        return True

    return False

def preprocess_image(url, size=(224, 224)):
    """
    Download and preprocess an image from a URL.
    Returns a resized PIL image in RGB mode if successful, otherwise None.
    """
    session = requests.Session()
    # Update headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': url,  # Simulate a referer header with the URL itself
    })
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Check if the image is empty
        if img.size == (0, 0):
            logger.error(f"Empty image at URL: {url}")
            return None

        # Check if the image is a placeholder
        if is_placeholder_image(img):
            logger.warning(f"Placeholder image detected at URL: {url}")
            return None

        # Resize the image to the specified size
        processed_img = img.resize(size)
        return processed_img

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            logger.warning(f"Image not found (404): {url}")
        else:
            logger.error(f"HTTP error for image {url}: {http_err}")
    except requests.RequestException as req_err:
        logger.error(f"Request error for image {url}: {req_err}")
    except Exception as e:
        logger.error(f"Error processing image {url}: {e}")

    return None


# class FLAVADataset(Dataset):
#     def __init__(self, dataframe, processor, label_type='2_way_label', include_metadata = True,  metadata_columns=['num_comments', 'score', 'upvote_ratio']):
#         self.dataframe = dataframe
#         self.processor = processor
#         self.label_type = label_type
#         self.include_metadata = include_metadata
#         self.metadata_columns = metadata_columns if include_metadata else []
       

#         # Ensure metadata is numeric
#         for col in self.metadata_columns:
#             self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')

#         # Calculate class weights
#         label_counts = self.dataframe[label_type].value_counts().sort_index()
#         class_weights = torch.tensor(label_counts.sum() / (len(label_counts) * label_counts), dtype=torch.float32)
#         self.class_weights = class_weights / class_weights.sum()

#         logging.info(f"Initialized FLAVADataset with {len(self.dataframe)} samples")

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]
#         text = row['clean_title']
#         image = row['image']
#         label = torch.tensor(row[self.label_type], dtype=torch.long)
#         metadata_values = row[self.metadata_columns].values.astype(float) if self.include_metadata else np.array([])  # Ensure metadata is float
        
#         try:
#             metadata = torch.tensor(metadata_values, dtype=torch.float32) if self.include_metadata else torch.tensor([], dtype=torch.float32)
#         except TypeError as e:
#             logging.error(f"Error converting metadata to tensor for row {idx}: {e}")
#             logging.error(f"Metadata values: {metadata_values}")
#             metadata = torch.tensor([0.0]*len(metadata_values), dtype=torch.float32) if self.include_metadata else torch.tensor([], dtype=torch.float32) # Default value in case of error

#         inputs = self.processor(
#             text=[text], images=image, return_tensors="pt", padding='max_length', truncation=True, max_length=40
#         )
#         inputs['labels'] = label.unsqueeze(0)  # Unsqueeze to make it compatible with batch dimension
#         if self.include_metadata: 
#             inputs['metadata'] = metadata  # Add metadata to inputs

#        # logging.info(f"Processed sample {idx}, label shape: {inputs['labels'].shape}, metadata shape: {inputs['metadata'].shape}")

#         return inputs

# # Custom collate function
# def collate_fn(batch):
#     batch = [item for item in batch if item['labels'].numel() > 0]  # Filter out empty labels
#     if len(batch) == 0:
#         return None  # Return None if the batch is empty

#     input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
#     attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
#     pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
#     labels = torch.cat([item['labels'].unsqueeze(0) if item['labels'].dim() == 0 else item['labels'] for item in batch], dim=0)
#     metadata = torch.stack([item['metadata'] for item in batch], dim=0) if 'metadata' in batch[0] else torch.tensor([], dtype=torch.float32)


#     #logging.info(f"Collated batch, input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")

#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'pixel_values': pixel_values,
#         'labels': labels.squeeze(),  # Make sure labels are of shape [batch_size]
#         'metadata': metadata if metadata.nelement() > 0 else None
#     }

# # Function to prepare data
# def get_flava_dataloader(df, processor, label_type, include_metadata=True):
#     dataset = FLAVADataset(df, processor, label_type, include_metadata)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
#     return dataloader, dataset.class_weights

# # Prepare data loaders for different label types
# print("Preparing data loaders...")
# data_loaders = {}
# class_weights = {}
# for label_type in ['2_way_label', '3_way_label']: #, '6_way_label'
#     train_features, cw = get_flava_dataloader(train_df, processor, label_type, INCLUDE_METADATA)
#     val_features, _ = get_flava_dataloader(val_df, processor, label_type, INCLUDE_METADATA)
#     test_features, _ = get_flava_dataloader(test_df, processor, label_type, INCLUDE_METADATA)
#     data_loaders[label_type] = {'train': train_features, 'val': val_features, 'test': test_features}
#     class_weights[label_type] = cw
# logging.info("Data loaders prepared.")