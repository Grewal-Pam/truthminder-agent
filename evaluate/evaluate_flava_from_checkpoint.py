import os
import torch
import pandas as pd
import numpy as np
from transformers import FlavaProcessor, FlavaModel
from models.flava_model import FlavaClassificationModel
from dataloaders.flava_dataloader import get_flava_dataloader
from evaluate.flava_evaluate import evaluate_flava
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
EXPERIMENT_NAME = "FLAVA/2025-04-09_18-08-22"
test_batch_file = "data/test_batch_1.tsv"
BATCH_SIZE = 16
DEVICE = "cpu"
LABEL_TYPES = {
    "2_way_label": 2,
    "3_way_label": 3
}
METADATA_COLUMNS = ["num_comments", "score", "upvote_ratio"]
# ----------------------

test_file_stem = os.path.splitext(os.path.basename(test_batch_file))[0]

print("[INFO] Loading test data...")
test_df = pd.read_csv(test_batch_file, sep="\t").sample(n=20, random_state=42)

print("[INFO] Loading FLAVA processor and model...")
flava_base = FlavaModel.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

def analyze_metadata_influence(model, test_df, processor, label_type, metadata_columns, experiment_name, device="cpu"):
    from dataloaders.flava_dataloader import get_flava_dataloader
    influence_scores = []
    valid_indices = []

    test_loader = get_flava_dataloader(
        test_df,
        processor,
        label_type=label_type,
        batch_size=1,
        include_metadata=True,
        metadata_columns=metadata_columns
    )

    model.eval()

    for batch in test_loader:
        if not isinstance(batch, dict):
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        metadata = batch['metadata'].to(device)

        with torch.no_grad():
            original_logits = model(input_ids, attention_mask, pixel_values, metadata)
            original_probs = torch.softmax(original_logits, dim=-1)

        row_influence = []
        for i in range(metadata.shape[1]):
            modified_metadata = metadata.clone()
            modified_metadata[0, i] = 0.0
            with torch.no_grad():
                modified_logits = model(input_ids, attention_mask, pixel_values, modified_metadata)
                modified_probs = torch.softmax(modified_logits, dim=-1)
            delta = torch.abs(original_probs - modified_probs).sum().item()
            row_influence.append(delta)

        influence_scores.append(row_influence)
        valid_indices.append(int(batch['row_index'].item()) if isinstance(batch['row_index'], torch.Tensor) else int(batch['row_index']))

    influence_array = np.array(influence_scores)
    row_df = pd.DataFrame(influence_array, columns=[f"inf_{col}" for col in metadata_columns])
    output_dir = os.path.join("runs", experiment_name, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    base_name = label_type.replace("_label", "")
    row_df.to_csv(os.path.join(output_dir, f"{test_file_stem}_{base_name}_metadata_influence_per_row.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(metadata_columns, influence_array.mean(axis=0), color='skyblue')
    plt.ylabel("Influence Score (Δ Softmax)")
    plt.title(f"Metadata Influence - {base_name.upper()}")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{test_file_stem}_{base_name}_metadata_influence.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT] Saved metadata influence plot to: {plot_path}")

    return row_df, valid_indices

# --- MAIN LOOP ---
for label_type, num_classes in LABEL_TYPES.items():
    for include_metadata in [True, False]:
        metadata_tag = "with_metadata" if include_metadata else "without_metadata"
        task_name = f"{label_type.replace('_label', '')}_classification"
        task_name_for_file = f"{label_type.replace('_label', '').replace('_', '-')}_classification"
        print(f"[RUN] Evaluating {task_name} ({metadata_tag})")

        test_loader = get_flava_dataloader(
            test_df,
            processor,
            label_type=label_type,
            batch_size=BATCH_SIZE,
            include_metadata=include_metadata,
            metadata_columns=METADATA_COLUMNS if include_metadata else None
        )

        checkpoint_name = f"{task_name_for_file}_{metadata_tag}_best_model.pth"
        checkpoint_path = os.path.join("runs", EXPERIMENT_NAME, "models", checkpoint_name)
        print(f"[LOAD] Loading model from: {checkpoint_path}")

        model = FlavaClassificationModel(
            flava_model=flava_base,
            num_labels=num_classes,
            metadata_dim=len(METADATA_COLUMNS) if include_metadata else 0,
            include_metadata=include_metadata
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        print("[INFO] Running evaluation...")
        metrics, (labels, preds, probs, val_losses, val_accuracies, row_indices) = evaluate_flava(
            model, test_loader, DEVICE, task_name=label_type.replace("_label", "")
        )

        row_indices = [int(idx) for idx in row_indices]
        print(f"[DEBUG] row_indices={row_indices}, test_df_len={len(test_df)}")
        if any(idx >= len(test_df) or idx < 0 for idx in row_indices):
            raise ValueError("Invalid row indices detected in evaluation output!")

        used_test_df = test_df.iloc[row_indices].reset_index(drop=True)
        if len(used_test_df) != len(labels):
            raise ValueError(f"[ERROR] Mismatch: used_test_df={len(used_test_df)}, labels={len(labels)}")

        if include_metadata:
            influence_df, influence_indices = analyze_metadata_influence(
                model=model,
                test_df=test_df,
                processor=processor,
                label_type=label_type,
                metadata_columns=METADATA_COLUMNS,
                experiment_name=EXPERIMENT_NAME,
                device=DEVICE
            )
        else:
            influence_df, influence_indices = None, None

        results_df = pd.DataFrame({"true_label": labels, "predicted_label": preds})
        probs_df = pd.DataFrame(np.array(probs), columns=[f"prob_class_{i}" for i in range(num_classes)])
        final_df = pd.concat([used_test_df.reset_index(drop=True), results_df, probs_df], axis=1)

        output_dir = os.path.join("runs", EXPERIMENT_NAME, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        detailed_path = os.path.join(output_dir, f"{test_file_stem}_{task_name_for_file}_{metadata_tag}_test_predictions_detailed.csv")
        final_df.to_csv(detailed_path, index=False)
        print(f"[SAVE] Predictions saved to: {detailed_path}")

        if influence_df is not None:
            if influence_indices == row_indices:
                merged_df = pd.concat([final_df.reset_index(drop=True), influence_df.reset_index(drop=True)], axis=1)
            else:
                print("[WARNING] Influence and prediction row index mismatch! Using shortest alignment.")
                min_len = min(len(final_df), len(influence_df))
                merged_df = pd.concat([
                    final_df.iloc[:min_len].reset_index(drop=True),
                    influence_df.iloc[:min_len].reset_index(drop=True)
                ], axis=1)

            combined_path = os.path.join(output_dir, f"{test_file_stem}_{task_name_for_file}_{metadata_tag}_test_predictions_combined.csv")
            merged_df.to_csv(combined_path, index=False)
            print(f"[SAVE] Combined predictions + influence saved to: {combined_path}")

        print(f"[RESULT] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
