import os
import json
import torch
import pandas as pd
from transformers import CLIPProcessor
from sklearn.metrics import classification_report
from models.clip_model import CLIPMultiTaskClassifier
from experiment_manager import prepare_dataloader
from training.evaluator import evaluate_model  # your custom evaluate method
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

# Patch the global experiment_logger used in experiment_manager
import experiment_manager

experiment_manager.experiment_logger = logging.getLogger("eval_logger")
experiment_manager.experiment_logger.setLevel(logging.WARNING)


# --- Setup ---
class Args:
    pass


args = Args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.batch_size = 16
args.input_dim = 512
args.num_classes_2 = 2
args.num_classes_3 = 3
args.metadata_columns = ["num_comments", "score", "upvote_ratio"]
args.labels_2 = [
    "0",
    "1",
]  # 2 way- 0 Fake,1 Real ::3 way- 0 Real, 1 MixedorSattire, 2 Fake
args.labels_3 = ["0", "1", "2"]
args.test_df = pd.read_csv("/home/ubuntu/projects/project/data/test_2000.csv")

# Base directory
model_dir = "/home/ubuntu/projects/project/runs/CLIP/2025-02-25_19-24-04/models"
output_dir = os.path.join(model_dir, "recalculated_metrics")
os.makedirs(output_dir, exist_ok=True)

# Processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Loop over metadata settings
for include_metadata in [True, False]:
    args.include_metadata = include_metadata
    metadata_tag = "with_metadata" if include_metadata else "without_metadata"

    print(f"\n🧪 Evaluating CLIP: {metadata_tag}")
    # Load test data
    test_loader = prepare_dataloader(
        dataframe=args.test_df,
        processor=processor,
        metadata_columns=args.metadata_columns,
        batch_size=args.batch_size,
        shuffle=False,
        include_metadata=include_metadata,
    )

    # Load model
    model = CLIPMultiTaskClassifier(
        input_dim=args.input_dim,
        num_classes_2=args.num_classes_2,
        num_classes_3=args.num_classes_3,
        metadata_dim=len(args.metadata_columns),
    ).to(args.device)

    # Load checkpoint
    ckpt_path = os.path.join(model_dir, f"clip_final_model_{metadata_tag}.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model.eval()

    # Evaluate
    _, _, test_labels_2, test_preds_2, _, test_labels_3, test_preds_3, _, _ = (
        evaluate_model(model=model, dataloader=test_loader, device=args.device)
    )

    # Classification reports
    report_2way = classification_report(
        test_labels_2, test_preds_2, output_dict=True, zero_division=0
    )
    report_3way = classification_report(
        test_labels_3, test_preds_3, output_dict=True, zero_division=0
    )

    # Save both
    with open(
        os.path.join(
            output_dir, f"clip_2-way_{metadata_tag}_REGENERATED_test_metrics.json"
        ),
        "w",
    ) as f:
        json.dump(report_2way, f, indent=2)

    with open(
        os.path.join(
            output_dir, f"clip_3-way_{metadata_tag}_REGENERATED_test_metrics.json"
        ),
        "w",
    ) as f:
        json.dump(report_3way, f, indent=2)

    print(f"✅ Saved: 2-way and 3-way classification reports for {metadata_tag}")
