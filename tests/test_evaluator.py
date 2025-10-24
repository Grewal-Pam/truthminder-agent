import pandas as pd
import pytest
import torch
from training.evaluator import evaluate_model
from models.clip_model import CLIPMultiTaskClassifier
from torch.utils.data import DataLoader
from local_datasets.pre_processing import compute_pixel_values, handle_missing_values, normalize_metadata

@pytest.fixture
def preprocessed_dataset():
    """
    Load and preprocess the test dataset.
    """
    test_path = "results/test_data.csv"
    df = pd.read_csv(test_path)

    # Ensure pixel values are computed
    df = compute_pixel_values(df, image_column="image_url")

    # Handle missing values
    columns_to_check = ["clean_title", "image_url", "num_comments", "score", "upvote_ratio", "2_way_label", "3_way_label", "pixel_values"]
    df = handle_missing_values(df, columns_to_check, method="drop")

    # Normalize metadata
    metadata_columns = ["num_comments", "score", "upvote_ratio"]
    if not all(col in df.columns for col in metadata_columns):
        raise KeyError(f"Missing metadata columns: {[col for col in metadata_columns if col not in df.columns]}")

    df = normalize_metadata(df, metadata_columns)

    # Combine metadata into a single column for the model
    df["metadata"] = df[metadata_columns].apply(lambda row: row.tolist(), axis=1)

    return df

class RealDataset(torch.utils.data.Dataset):
    """
    Dataset class for the preprocessed dataset.
    """
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Ensure pixel values are properly handled
        pixel_values = row["pixel_values"]
        if isinstance(pixel_values, str):
            pixel_values = eval(pixel_values)  # Convert stringified list to actual list
        elif isinstance(pixel_values, list):
            pixel_values = pixel_values
        else:
            raise TypeError(f"Unexpected pixel_values type: {type(pixel_values)}")

        return {
            "input_ids": torch.tensor(eval(row["input_ids"]), dtype=torch.long),
            "attention_mask": torch.tensor(eval(row["attention_mask"]), dtype=torch.long),
            "pixel_values": torch.tensor(pixel_values, dtype=torch.float),
            "labels_2_way": torch.tensor(row["2_way_label"], dtype=torch.long),
            "labels_3_way": torch.tensor(row["3_way_label"], dtype=torch.long),
            "metadata": torch.tensor(row["metadata"], dtype=torch.float),
        }

@pytest.fixture
def dataloader(preprocessed_dataset):
    """
    DataLoader for the preprocessed test dataset.
    """
    dataset = RealDataset(preprocessed_dataset)
    return DataLoader(dataset, batch_size=16, shuffle=False)

@pytest.fixture
def trained_model():
    """
    Load the trained model with saved weights.
    """
    model = CLIPMultiTaskClassifier(input_dim=512, num_classes_2=2, num_classes_3=3, metadata_dim=3)
    model.load_state_dict(torch.load("results/model_weights.pth"))
    return model

def test_evaluate_model(trained_model, dataloader):
    """
    Test the evaluation process using the preprocessed dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    # Run evaluation
    avg_loss, metrics = evaluate_model(trained_model, dataloader, device)

    # Assert the average loss is a float
    assert isinstance(avg_loss, float), "Average loss should be a float."

    # Check metrics for both tasks
    for task in ["2_way", "3_way"]:
        assert "accuracy" in metrics[task], f"Accuracy missing for {task}"
        assert "precision" in metrics[task], f"Precision missing for {task}"
        assert "recall" in metrics[task], f"Recall missing for {task}"
        assert "f1" in metrics[task], f"F1-score missing for {task}"
        assert "kappa" in metrics[task], f"Kappa missing for {task}"
        assert "confusion_matrix" in metrics[task], f"Confusion matrix missing for {task}"

    print("Evaluation test passed for the preprocessed dataset.")
