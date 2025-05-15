import pytest
import pandas as pd
import torch
from datasets.dataset import load_images
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from unittest.mock import patch
from main import run_experiment
from datasets.data_analysis import DataAnalysis
from datasets.pre_processing import normalize_metadata
from models.clip_model import CLIPMultiTaskClassifier
from training.trainer import train_model
from training.evaluator import evaluate_model
from utils.plotting import Plotting


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "clean_title": [
            "My Walgreens offbrand Mucinex was engraved with the letters Mucinex but in a different order",
            "Hackers leak emails from UAE ambassador to US",
            "Puppy taking in the view",
            "Bride and groom exchange vows after fatal shooting at their wedding"
        ],
        "image_url": [
            "https://external-preview.redd.it/WylDbZrnbvZdBpgfa3ntxYf17CBHndiJWHylVm2j_nY.jpg?width=320&crop=smart&auto=webp&s=449659a10792de4d55c2f27d2176fdc8bc66e72a",
            "https://external-preview.redd.it/6fNhdbc6K1vFAZX95ed-ZMLDlASqmpixvAxmXqHhfHs.jpg?width=320&crop=smart&auto=webp&s=f263f843e91b7dbea1d1157031f7ea0d555186ed",
            "https://external-preview.redd.it/HLtVNhTR6wtYtnvL8RKfmZFdKQ7tVz42WCiH3jSGiIk.jpg?width=320&crop=smart&auto=webp&s=2d60c452b15d7d86b46b65e06da9a6e42f10d5df",
            "https://external-preview.redd.it/FQ-J9OIPFRpqi912TpCYelw2HUtsIiJPJIny94M39kw.jpg?width=320&crop=smart&auto=webp&s=28a3ac355c7ff188f3622a11c488cb8d36e0e488"
        ],
        "2_way_label": [0, 1, 1, 0],
        "3_way_label": [0, 1, 2, 1],
        "num_comments": [12, 44, 250, 6],
        "score": [2.0, 1.0, 26.0, 7.0],
        "upvote_ratio": [0.84, 0.92, 0.95, 0.64]
    })




def test_training_and_evaluation(sample_data):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    class GeneralDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, processor):
            self.dataframe = dataframe
            self.processor = processor

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            inputs = {
                "input_ids": torch.randint(0, 999, (40,)),
                "attention_mask": torch.ones(40),
                "pixel_values": torch.randn(3, 224, 224),
                "labels_2_way": torch.tensor(row["2_way_label"]),
                "labels_3_way": torch.tensor(row["3_way_label"]),
                "metadata": torch.tensor([row[col] for col in ["num_comments", "score", "upvote_ratio"]], dtype=torch.float32)
            }
            return inputs

    dataset = GeneralDataset(sample_data, processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CLIPMultiTaskClassifier(input_dim=512, num_classes_2=2, num_classes_3=3, metadata_dim=3)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


     # Training
    train_loss = train_model(model, dataloader, optimizer, device=torch.device("cpu"))
    assert train_loss > 0, "Training loss should be greater than 0"

    # Evaluation
    val_loss, metrics = evaluate_model(model, dataloader, device=torch.device("cpu"))
    assert "2_way" in metrics
    assert "3_way" in metrics
    assert metrics["2_way"]["accuracy"] >= 0.0
    assert metrics["3_way"]["accuracy"] >= 0.0



@patch("main.save_metrics")
def test_run_experiment(mock_save_metrics):
    """
    Test the main experiment pipeline.
    """
    mock_save_metrics.return_value = None  # Mock save_metrics to skip actual file writing

    try:
        # Run the main experiment
        run_experiment()
    except Exception as e:
        # Add debug logs
        print("==== Debug Information ====")
        print("Exception:", e)
        
        # If you want to debug the model, you need to define or import it here
        from models.clip_model import CLIPMultiTaskClassifier
        model = CLIPMultiTaskClassifier(
            input_dim=512,
            num_classes_2=2,
            num_classes_3=3,
            metadata_dim=3,
        )
        for name, param in model.named_parameters():
            print(f"{name}: {param.data}")

        # Raise the exception again for test failure
        raise

    # Verify save_metrics was called
    mock_save_metrics.assert_called()

def test_data_analysis(sample_dataframe):
    data_analysis = DataAnalysis()
    data_analysis.check_missing_values(sample_dataframe)
    data_analysis.normalize_metadata(sample_dataframe, ["num_comments", "score"])
    data_analysis.check_outliers(sample_dataframe, ["num_comments", "score"])
    data_analysis.check_class_imbalance(sample_dataframe, "2_way_label")

def test_plotting(sample_dataframe, tmp_path):
    plotter = Plotting(output_dir=tmp_path)
    plotter.plot_histogram(sample_dataframe, "num_comments")
    plotter.plot_class_distribution(sample_dataframe, "2_way_label")

    # Verify plots are saved
    assert (tmp_path / "histogram_num_comments.png").exists()
    assert (tmp_path / "class_distribution_2_way_label.png").exists()

@pytest.fixture
def model_fixture():
    return CLIPMultiTaskClassifier(input_dim=512, num_classes_2=2, num_classes_3=3, metadata_dim=3)

def test_model_saving_and_loading(model_fixture, tmp_path):
    model = model_fixture
    model_path = tmp_path / "test_model.pt"

    # Save the model
    torch.save(model.state_dict(), model_path)

    # Check if the file exists
    assert model_path.exists(), "Model file not saved correctly."

    # Load the model
    loaded_model = CLIPMultiTaskClassifier(input_dim=512, num_classes_2=2, num_classes_3=3, metadata_dim=3)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    # Verify the loaded model parameters match the original
    for param_original, param_loaded in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param_original, param_loaded), "Model parameters do not match after loading."

    # Run a forward pass with dummy inputs
    input_ids = torch.randint(0, 512, (1, 40))
    attention_mask = torch.ones((1, 40))
    pixel_values = torch.randn((1, 3, 224, 224))
    metadata = torch.randn((1, 3))

    try:
        outputs = loaded_model(input_ids, attention_mask, pixel_values, metadata)
        assert outputs is not None, "Loaded model did not produce outputs."
    except Exception as e:
        pytest.fail(f"Loaded model forward pass failed: {e}")
