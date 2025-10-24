import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from transformers import FlavaProcessor, FlavaModel

from models.flava_model import FlavaClassificationModel
from dataloaders.flava_dataloader import get_flava_dataloader
from evaluate.flava_evaluate import evaluate_flava
from utils.logger import setup_logger

logger = setup_logger("flava_eval_logger")


# ---------- Helper for Metadata Influence ----------
def analyze_metadata_influence(
    model, test_df, processor, label_type, metadata_columns, device, batch_size=1
):
    influence_scores = []
    valid_indices = []

    test_loader = get_flava_dataloader(
        df=test_df,
        processor=processor,
        label_type=label_type,
        batch_size=batch_size,
        include_metadata=True,
        metadata_columns=metadata_columns,
    )

    model.eval()
    for batch in test_loader:
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        metadata = batch["metadata"].to(device)

        with torch.no_grad():
            original_probs = torch.softmax(
                model(input_ids, attention_mask, pixel_values, metadata), dim=-1
            )

        row_influence = []
        for i in range(metadata.shape[1]):
            modified = metadata.clone()
            modified[0, i] = 0.0
            with torch.no_grad():
                new_probs = torch.softmax(
                    model(input_ids, attention_mask, pixel_values, modified), dim=-1
                )
            delta = torch.abs(original_probs - new_probs).sum().item()
            row_influence.append(delta)

        influence_scores.append(row_influence)
        valid_indices.append(int(batch["row_index"].item()))

    return (
        pd.DataFrame(
            influence_scores, columns=[f"inf_{col}" for col in metadata_columns]
        ),
        valid_indices,
    )


# ---------- Args ----------
class Args:
    pass


args = Args()
args.metadata_columns = ["num_comments", "score", "upvote_ratio"]
args.batch_size = 16
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.test_df = pd.read_csv("/home/ubuntu/projects/project/data/test_2000.csv")
args.labels_2 = ["0", "1"]
args.labels_3 = ["0", "1", "2"]

# ---------- Paths ----------
base_model_dir = "/home/ubuntu/projects/project/runs/FLAVA/2025-04-09_18-08-22/models"
output_dir = os.path.join(base_model_dir, "recalculated_metrics")
os.makedirs(output_dir, exist_ok=True)

# ---------- Config ----------
tasks = [
    ("2-way classification", "2_way_label", 2, args.labels_2),
    ("3-way classification", "3_way_label", 3, args.labels_3),
]
metadata_flags = [True, False]

# ---------- Processor ----------
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

# ---------- Main Loop ----------
for include_metadata in metadata_flags:
    args.include_metadata = include_metadata
    metadata_tag = "with_metadata" if include_metadata else "without_metadata"

    for task_name, label_col, num_labels, class_labels in tasks:
        print(f"\n🧪 Evaluating: {task_name} | {metadata_tag}")
        label_counts = Counter(args.test_df[label_col].tolist())
        logger.info(f"[{task_name}] Test label distribution: {label_counts}")

        # Load dataloader
        test_loader = get_flava_dataloader(
            df=args.test_df,
            processor=processor,
            label_type=label_col,
            batch_size=args.batch_size,
            include_metadata=include_metadata,
            metadata_columns=args.metadata_columns if include_metadata else None,
        )

        # Load model
        flava_base = FlavaModel.from_pretrained("facebook/flava-full")
        model = FlavaClassificationModel(
            flava_base,
            num_labels=num_labels,
            metadata_dim=len(args.metadata_columns) if include_metadata else 0,
            include_metadata=include_metadata,
        ).to(args.device)

        model_ckpt = os.path.join(
            base_model_dir,
            f"{task_name.replace(' ', '_')}_{metadata_tag}_best_model.pth",
        )
        if not os.path.exists(model_ckpt):
            print(f"❌ Model not found: {model_ckpt}")
            continue
        model.load_state_dict(torch.load(model_ckpt, map_location=args.device))
        model.eval()

        # Evaluate and Save Metrics
        metrics, _ = evaluate_flava(model, test_loader, args.device)
        output_path = os.path.join(
            output_dir,
            f"{task_name.replace(' ', '_')}_{metadata_tag}_REGENERATED_test_metrics.json",
        )
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Saved: {output_path}")

        # 🔍 Extended Outputs if metadata enabled
        if include_metadata:
            _, (labels, preds, probs, _, _, row_indices) = evaluate_flava(
                model, test_loader, args.device
            )
            used_test_df = args.test_df.iloc[row_indices].reset_index(drop=True)
            results_df = pd.DataFrame({"true_label": labels, "predicted_label": preds})
            probs_df = pd.DataFrame(
                probs, columns=[f"prob_class_{i}" for i in range(num_labels)]
            )
            final_df = pd.concat(
                [used_test_df.reset_index(drop=True), results_df, probs_df], axis=1
            )

            # Compute metadata influence
            influence_df, influence_indices = analyze_metadata_influence(
                model=model,
                test_df=args.test_df,
                processor=processor,
                label_type=label_col,
                metadata_columns=args.metadata_columns,
                device=args.device,
            )

            # Merge and Save CSV
            if influence_indices == row_indices:
                combined_df = pd.concat(
                    [
                        final_df.reset_index(drop=True),
                        influence_df.reset_index(drop=True),
                    ],
                    axis=1,
                )
            else:
                logger.warning("Mismatch between prediction and influence indices.")
                min_len = min(len(final_df), len(influence_df))
                combined_df = pd.concat(
                    [
                        final_df.iloc[:min_len].reset_index(drop=True),
                        influence_df.iloc[:min_len].reset_index(drop=True),
                    ],
                    axis=1,
                )

            combined_path = os.path.join(
                output_dir,
                f"{task_name.replace(' ', '_')}_{metadata_tag}_test_predictions_combined.csv",
            )
            combined_df.to_csv(combined_path, index=False)
            logger.info(
                f"✅ Combined predictions + influence saved to: {combined_path}"
            )

            # Save bar plot
            plt.figure(figsize=(8, 4))
            plt.bar(args.metadata_columns, influence_df.mean(axis=0), color="teal")
            plt.ylabel("Influence Score (Δ Softmax)")
            plt.title(f"FLAVA Metadata Influence - {task_name} ({metadata_tag})")
            plt.tight_layout()
            plot_path = os.path.join(
                output_dir,
                f"{task_name.replace(' ', '_')}_{metadata_tag}_influence.png",
            )
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"📊 Influence plot saved: {plot_path}")
