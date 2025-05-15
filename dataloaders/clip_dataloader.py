import os
import requests
import config
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor
from utils.logger import setup_logger

logger = setup_logger("clip_dataloader", log_dir="runs/logs", sampled=False )#log_dir=config.log_dir, sampled=True


def get_clip_dataloader(dataframe, processor, metadata_columns, batch_size, shuffle=False):
    """
    Prepare a DataLoader for CLIP training and evaluation.

    Args:
        dataframe (pd.DataFrame): Input dataset.
        processor (CLIPProcessor): CLIP processor for tokenization and image processing.
        metadata_columns (list): List of column names for metadata features.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Torch DataLoader with processed data.
    """
    class ClipDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, processor, label_type='2_way_label', include_metadata=True, metadata_columns=None):
            self.dataframe = dataframe
            self.processor = processor
            self.label_type = label_type
            self.include_metadata = include_metadata
            self.metadata_columns = metadata_columns if metadata_columns else []


        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            text = row["clean_title"]
            image_path = row["image_path"]
            metadata_values = (
                row[self.metadata_columns].values.astype(float)
                if self.include_metadata else []
            )
            labels_2 = row["2_way_label"]
            labels_3 = row["3_way_label"]

            # Process text and image
            inputs = self.processor(
                text=[text],
                images=Image.open(image_path).convert("RGB"),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=40
            )
            inputs["metadata"] = torch.tensor(metadata_values, dtype=torch.float32)
            inputs["labels_2_way"] = torch.tensor(labels_2, dtype=torch.long)
            inputs["labels_3_way"] = torch.tensor(labels_3, dtype=torch.long)

            # Remove batch dimension from processor outputs
            return {key: val.squeeze(0) for key, val in inputs.items()}

    # Create dataset and DataLoader
    dataset = ClipDataset(dataframe, processor, metadata_columns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
