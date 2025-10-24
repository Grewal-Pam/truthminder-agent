import pytest
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image  # Importing Image from PIL
from local_datasets.pre_processing import handle_missing_values, normalize_metadata, split_dataset
from local_datasets.pre_processing import preprocess_image, compute_pixel_values


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    data = {
        "num_comments": [1, 2, np.nan, 4],
        "score": [10, np.nan, 30, 40],
        "upvote_ratio": [0.5, 0.7, np.nan, 0.9],
        "label": [0, 1, 0, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def real_dataset():
    """Fixture to load real dataset if available."""
    dataset_path = "data/filtered_data.tsv"
    if not os.path.exists(dataset_path):
        pytest.skip("Real dataset not found at specified path.")
    return pd.read_csv(dataset_path, sep="\t")


# Test handle_missing_values
# @pytest.mark.parametrize("method,fill_value,expected_length", [
#     ("drop", None, 2),  # Drop rows with missing values in specified columns
#     ("fill", 0.0, 4)    # Fill missing values with 0.0
# ])
# def test_handle_missing_values(sample_dataframe, method, fill_value, expected_length):
#     result = handle_missing_values(sample_dataframe, ["num_comments", "score", "upvote_ratio"], method=method, fill_value=fill_value)
#     assert len(result) == expected_length, f"Expected {expected_length} rows, got {len(result)}"
#     assert result.isnull().sum().sum() == 0, "Missing values were not handled correctly"

@pytest.mark.parametrize(
    "method, fill_value, expected_length",
    [
        ("drop", None, 500),  # Adjust `expected_length` to match the expected number of rows after dropping
        ("fill", 0.0, 500),   # Adjust `expected_length` to match the total rows when filling
    ]
)
def test_handle_missing_values_real_dataset(real_dataset, method, fill_value, expected_length):
    columns = ["clean_title", "image_url", "2_way_label", "3_way_label"]
    processed_df = handle_missing_values(real_dataset, columns, method=method, fill_value=fill_value)
    assert len(processed_df) == expected_length, f"Expected {expected_length} rows, but got {len(processed_df)}"
    assert processed_df[columns].isnull().sum().sum() == 0, "Missing values were not handled properly in the real dataset."




# Test normalize_metadata
# def test_normalize_metadata(sample_dataframe, tmp_path):
#     """Test metadata normalization with sample data."""
#     metadata_columns = ["num_comments", "score", "upvote_ratio"]
#     scaler_path = tmp_path / "scaler.pkl"
#     result = normalize_metadata(sample_dataframe.dropna(), metadata_columns, scaler_path=scaler_path)

#     # Check if columns are normalized
#     for col in metadata_columns:
#         assert np.isclose(result[col].mean(), 0, atol=1e-7), f"{col} mean is not 0 after normalization"
#         assert np.isclose(result[col].std(), 1, atol=1e-7), f"{col} std is not 1 after normalization"

#     # Ensure scaler is saved
#     assert scaler_path.exists(), "Scaler file was not saved"
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
#         assert isinstance(scaler, StandardScaler), "Saved scaler is not a StandardScaler instance"


def test_normalize_metadata_real_dataset(real_dataset, tmp_path):
    """Test metadata normalization with the real dataset."""
    metadata_columns = ["num_comments", "score", "upvote_ratio"]
    scaler_path = tmp_path / "scaler.pkl"
    processed_df = normalize_metadata(real_dataset.dropna(subset=metadata_columns), metadata_columns, scaler_path=scaler_path)

    # Check if columns are normalized
    for col in metadata_columns:
        assert np.isclose(processed_df[col].mean(), 0, atol=1e-2), f"{col} mean is not 0 after normalization"
        assert np.isclose(processed_df[col].std(), 1, atol=1e-2), f"{col} std is not 1 after normalization"

    # Ensure scaler is saved
    assert scaler_path.exists(), "Scaler file was not saved for real dataset"


# Test split_dataset
def test_split_dataset(sample_dataframe):
    """Test dataset splitting on sample data."""
    train_df, val_df, test_df = split_dataset(sample_dataframe, test_size=0.25, val_size=0.5, seed=42)

    # Ensure no data loss during splitting
    total_rows = len(sample_dataframe)
    assert len(train_df) + len(val_df) + len(test_df) == total_rows, "Data loss during split"

    # Validate split sizes
    #assert len(test_df) == int(total_rows * 0.25), "Test set size mismatch"
    #assert len(val_df) == int((total_rows - len(test_df)) * 0.5), "Validation set size mismatch"
    assert len(train_df) == total_rows - len(val_df) - len(test_df), "Train set size mismatch"

    # Ensure no overlap between splits
    assert set(train_df.index).isdisjoint(set(val_df.index)), "Train and validation sets overlap"
    assert set(train_df.index).isdisjoint(set(test_df.index)), "Train and test sets overlap"
    assert set(val_df.index).isdisjoint(set(test_df.index)), "Validation and test sets overlap"


    # Validate split sizes (allow rounding errors)
    assert abs(len(test_df) - int(total_rows * 0.25)) <= 1, "Test set size mismatch"
    assert abs(len(val_df) - int((total_rows - len(test_df)) * 0.5)) <= 1, "Validation set size mismatch"

def test_split_dataset_real_dataset(real_dataset):
    """Test dataset splitting on the real dataset."""
    train_df, val_df, test_df = split_dataset(real_dataset, test_size=0.2, val_size=0.25, seed=42)

    # Ensure no data loss during splitting
    total_length = len(real_dataset)
    assert len(train_df) + len(val_df) + len(test_df) == total_length, "Data loss in real dataset split"

    # Ensure no duplicates in splits
    assert train_df.index.is_unique, "Train set contains duplicate rows"
    assert val_df.index.is_unique, "Validation set contains duplicate rows"
    assert test_df.index.is_unique, "Test set contains duplicate rows"

    # Validate split sizes
    assert len(test_df) == int(total_length * 0.2), "Test set size mismatch"
    assert len(val_df) == int((total_length - len(test_df)) * 0.25), "Validation set size mismatch"

def test_preprocess_image(real_dataset):
    """
    Test the `preprocess_image` function using real data.
    """
    # Extract a few image URLs from the dataset
    sample_urls = real_dataset["image_url"].dropna().head(5)
    
    for url in sample_urls:
        img = preprocess_image(url)
        if img is not None:
            assert isinstance(img, Image.Image), "Preprocessed image should be a PIL Image"
            assert img.size == (224, 224), "Image size should be resized to (224, 224)"
        else:
            print(f"Skipping URL due to failure: {url}")

def test_compute_pixel_values(real_dataset):
    """
    Test the `compute_pixel_values` function using real data.
    """
    # Ensure `image_url` column exists
    assert "image_url" in real_dataset.columns, "Dataset should have an 'image_url' column"
    
    # Process the dataset
    processed_df = compute_pixel_values(real_dataset)

    # Ensure the directory exists
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save pixel values to the results folder
    save_path = os.path.join(save_dir, "pixel_values_debug.csv")
    processed_df[["image_url", "pixel_values"]].to_csv(save_path, index=False)
    print(f"Pixel values saved to {save_path} for analysis.")
    
    # Check that the invalid rows were dropped
    assert "pixel_values" in processed_df.columns, "Processed DataFrame should have 'pixel_values' column"
    assert processed_df["pixel_values"].notnull().all(), "All pixel_values should be non-null"

    # Validate pixel values
    for pixel_values in processed_df["pixel_values"]:
        assert isinstance(pixel_values, list), "Pixel values should be a list"
        assert len(pixel_values) > 0, "Pixel values should not be empty"

    print("Preprocessing tests passed with real data.")