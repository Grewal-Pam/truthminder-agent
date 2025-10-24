import pytest
import torch
import pandas as pd
from transformers import AutoTokenizer
from models.clip_model import CLIPMultiTaskClassifier
from local_datasets.pre_processing import handle_missing_values, normalize_metadata, split_dataset

@pytest.fixture
def preprocessed_data():
    """
    Preprocess the dataset and return train, validation, and test splits.
    """
    data_path = "data/filtered_data.tsv"
    df = pd.read_csv(data_path, sep="\t")

    # Handle missing values
    columns_to_check = ["clean_title", "image_url", "num_comments", "score", "upvote_ratio", "2_way_label", "3_way_label"]
    df = handle_missing_values(df, columns_to_check, method="drop")

    # Normalize metadata columns
    metadata_columns = ["num_comments", "score", "upvote_ratio"]
    df = normalize_metadata(df, metadata_columns, scaler_path="scaler.pkl")

    # Split dataset
    train_df, val_df, test_df = split_dataset(df, test_size=0.2, val_size=0.25, seed=42)
    return train_df, val_df, test_df

@pytest.fixture
def real_data_inputs(preprocessed_data):
    """
    Prepare inputs from the preprocessed dataset for testing the model.
    """
    train_df, _, _ = preprocessed_data

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Tokenize `clean_title` column
    tokenized = tokenizer(list(train_df["clean_title"]), padding="max_length", truncation=True, max_length=10, return_tensors="pt")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    batch_size = input_ids.shape[0]

    # Prepare metadata and pixel values
    metadata = torch.tensor(train_df[["num_comments", "score", "upvote_ratio"]].values[:batch_size], dtype=torch.float)
    pixel_values = torch.randn((batch_size, 3, 224, 224))  # Placeholder for image inputs

    return input_ids, attention_mask, pixel_values, metadata

def test_clip_model_forward(real_data_inputs):
    """
    Test the forward pass of the CLIPMultiTaskClassifier model.
    """
    input_dim = 512
    metadata_dim = 3
    num_classes_2 = 2
    num_classes_3 = 3

    # Initialize the model
    model = CLIPMultiTaskClassifier(input_dim, num_classes_2, num_classes_3, metadata_dim)

    # Forward pass
    input_ids, attention_mask, pixel_values, metadata = real_data_inputs
    outputs = model(input_ids, attention_mask, pixel_values, metadata)

    # Check output keys
    assert "2_way" in outputs, "Output should have '2_way' key."
    assert "3_way" in outputs, "Output should have '3_way' key."

    # Check output shapes
    assert outputs["2_way"].shape == (input_ids.shape[0], num_classes_2), \
        f"Expected shape {(input_ids.shape[0], num_classes_2)}, got {outputs['2_way'].shape}."
    assert outputs["3_way"].shape == (input_ids.shape[0], num_classes_3), \
        f"Expected shape {(input_ids.shape[0], num_classes_3)}, got {outputs['3_way'].shape}."

def test_combined_representation(real_data_inputs):
    """
    Test the combined representation of text and metadata features.
    """
    input_dim = 512
    metadata_dim = 3
    num_classes_2 = 2
    num_classes_3 = 3

    # Initialize the model
    model = CLIPMultiTaskClassifier(input_dim, num_classes_2, num_classes_3, metadata_dim)

    # Forward pass
    input_ids, attention_mask, pixel_values, metadata = real_data_inputs
    outputs = model(input_ids, attention_mask, pixel_values, metadata)

    # Check combined representation's functionality
    combined_representation = model.text_fc(model.embedding(input_ids).mean(dim=1)) + model.metadata_fc(metadata)

    assert combined_representation.shape == (input_ids.shape[0], input_dim), \
        f"Expected shape {(input_ids.shape[0], input_dim)}, got {combined_representation.shape}."

def test_embedding_layer(real_data_inputs):
    """
    Test the embedding layer's output shape.
    """
    input_dim = 512
    metadata_dim = 3
    num_classes_2 = 2
    num_classes_3 = 3

    # Initialize the model
    model = CLIPMultiTaskClassifier(input_dim, num_classes_2, num_classes_3, metadata_dim)

    # Check embedding output shape
    input_ids, _, _, _ = real_data_inputs
    embedded_ids = model.embedding(input_ids)
    assert embedded_ids.shape == (input_ids.shape[0], input_ids.shape[1], input_dim), \
        f"Expected shape {(input_ids.shape[0], input_ids.shape[1], input_dim)}, got {embedded_ids.shape}."
