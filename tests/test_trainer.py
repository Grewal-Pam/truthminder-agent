import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer
from models.clip_model import CLIPMultiTaskClassifier
from training.trainer import train_model
from local_datasets.pre_processing import handle_missing_values, normalize_metadata, split_dataset, compute_pixel_values
import pandas as pd
import os
from utils.logger import setup_logger

# Step 1: Load and preprocess dataset
data_path = "data/filtered_data.tsv"
df = pd.read_csv(data_path, sep="\t")

# Compute pixel values
df = compute_pixel_values(df, image_column="image_url")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize the `clean_title` column
tokenized = tokenizer(df["clean_title"].tolist(), padding="max_length", truncation=True, max_length=10, return_tensors="pt")

# Add `input_ids` and `attention_mask` to DataFrame
df["input_ids"] = tokenized["input_ids"].tolist()
df["attention_mask"] = tokenized["attention_mask"].tolist()

# Handle missing values
columns_to_check = ["clean_title", "image_url", "num_comments", "score", "upvote_ratio", "2_way_label", "3_way_label", "pixel_values"]
df = handle_missing_values(df, columns_to_check, method="drop")

# Normalize metadata
metadata_columns = ["num_comments", "score", "upvote_ratio"]
df = normalize_metadata(df, metadata_columns)

# Split dataset
train_df, val_df, test_df = split_dataset(df, test_size=0.2, val_size=0.25, seed=42)

# Step 2: Create Dataset and DataLoader
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "pixel_values": torch.tensor(row["pixel_values"], dtype=torch.float),
            "metadata": torch.tensor([row[col] for col in metadata_columns], dtype=torch.float),
            "labels_2_way": torch.tensor(row["2_way_label"], dtype=torch.long),
            "labels_3_way": torch.tensor(row["3_way_label"], dtype=torch.long),
        }

# Create DataLoader
train_dataset = CustomDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 3: Initialize Model
input_dim = 512
num_classes_2 = 2
num_classes_3 = 3
metadata_dim = len(metadata_columns)

model = CLIPMultiTaskClassifier(input_dim, num_classes_2, num_classes_3, metadata_dim)

# Step 4: Optimizer and Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)

# Step 5: Train the Model
average_loss = train_model(model, train_loader, optimizer, device, clip_grad=True)

# Step 6: Validate Pixel Value Outputs

