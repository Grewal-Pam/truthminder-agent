# evaluate_clip_from_checkpoint.py

import os
import torch
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from models.clip_model import CLIPMultiTaskClassifier
import torch.nn.functional as F
import matplotlib.pyplot as plt
from experiment_manager import prepare_dataloader


# --- CONFIGURATION ---
EXPERIMENT_NAME = "CLIP/2025-04-13_07-20-33"
test_batch_file = "data/test_batch_1.tsv"
BATCH_SIZE = 16
DEVICE = "cpu"
LABEL_TYPES = {"2_way_label": 2, "3_way_label": 3}
METADATA_COLUMNS = ["num_comments", "score", "upvote_ratio"]
INPUT_DIM = 512
# ----------------------

print("[INFO] Loading test data...")
test_df = pd.read_csv(test_batch_file, sep="\t").sample(n=20, random_state=42)

print("[INFO] Loading CLIP processor...")
clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def analyze_metadata_influence(
    model,
    test_df,
    processor,
    label_type,
    metadata_columns,
    experiment_name,
    device="cpu",
):
    print("[INFO] Estimating metadata feature influence for CLIP...")

    from experiment_manager import prepare_dataloader  # Use CLIP’s own loader

    influence_scores = []

    test_loader = prepare_dataloader(
        dataframe=test_df,
        processor=processor,
        metadata_columns=metadata_columns,
        batch_size=1,
        shuffle=False,
        include_metadata=True,
    )

    model.eval()

    for batch in test_loader:
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        metadata = batch["metadata"].to(device)

        label_key = "2_way" if label_type == "2_way_label" else "3_way"

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, pixel_values, metadata)
            original_probs = F.softmax(outputs[label_key], dim=-1)

        row_influence = []
        for i in range(metadata.shape[1]):
            modified_metadata = metadata.clone()
            modified_metadata[0, i] = 0.0

            with torch.no_grad():
                outputs_mod = model(
                    input_ids, attention_mask, pixel_values, modified_metadata
                )
                modified_probs = F.softmax(outputs_mod[label_key], dim=-1)

            delta = torch.abs(original_probs - modified_probs).sum().item()
            row_influence.append(delta)

        influence_scores.append(row_influence)

    influence_array = np.array(influence_scores)
    row_df = pd.DataFrame(
        influence_array, columns=[f"inf_{col}" for col in metadata_columns]
    )
    output_dir = os.path.join("runs", experiment_name, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    base_name = label_type.replace("_label", "")
    row_df.to_csv(
        os.path.join(output_dir, f"{base_name}_metadata_influence_per_row.csv"),
        index=False,
    )

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(metadata_columns, influence_array.mean(axis=0), color="coral")
    plt.ylabel("Influence Score (Δ Softmax)")
    plt.title(f"CLIP Metadata Influence - {base_name.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_metadata_influence.png"))
    plt.close()

    print("[INFO] Metadata influence plot saved.")
    return row_df


# --- MAIN LOOP ---
for label_type, num_classes in LABEL_TYPES.items():
    include_metadata = True  # only evaluate the single available model
    task_name = f"{label_type.replace('_label', '').replace('_', '-')}_classification"
    checkpoint_path = os.path.join(
        "runs", EXPERIMENT_NAME, "models", "clip_best_model_with_metadata.pth"
    )

    print(f"[LOAD] Loading model: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model = CLIPMultiTaskClassifier(
        INPUT_DIM, 2, 3, len(METADATA_COLUMNS), include_metadata=True
    )
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print(f"[EVAL] Evaluating: {task_name} (with_metadata)")
    test_loader = prepare_dataloader(
        dataframe=test_df,
        processor=processor,
        metadata_columns=METADATA_COLUMNS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        include_metadata=True,
    )

    labels, preds, probs, row_indices = [], [], [], []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values = batch["pixel_values"].to(DEVICE)
        metadata = batch["metadata"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, pixel_values, metadata)
            # logits = outputs[label_type.replace("_label", "")]
            logits = (
                outputs["2_way"] if label_type == "2_way_label" else outputs["3_way"]
            )

            probabilities = F.softmax(logits, dim=-1)
            preds.extend(torch.argmax(probabilities, dim=1).tolist())
            probs.extend(probabilities.tolist())
            # labels.extend(batch[label_type].tolist())
            label_key = (
                "labels_2_way" if label_type == "2_way_label" else "labels_3_way"
            )
            labels.extend(batch[label_key].tolist())

            row_indices.extend(batch["row_index"].tolist())

    used_test_df = test_df.iloc[row_indices].reset_index(drop=True)
    results_df = pd.DataFrame({"true_label": labels, "predicted_label": preds})
    probs_df = pd.DataFrame(
        probs, columns=[f"prob_class_{i}" for i in range(num_classes)]
    )
    final_df = pd.concat([used_test_df, results_df, probs_df], axis=1)

    output_dir = os.path.join("runs", EXPERIMENT_NAME, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(
        output_dir, f"{task_name}_with_metadata_test_predictions_detailed.csv"
    )
    final_df.to_csv(detailed_path, index=False)
    print(f"[SAVE] Saved predictions to: {detailed_path}")

    influence_df = analyze_metadata_influence(
        model, test_df, processor, label_type, METADATA_COLUMNS, EXPERIMENT_NAME, DEVICE
    )
    merged_df = pd.concat(
        [final_df.reset_index(drop=True), influence_df.reset_index(drop=True)], axis=1
    )
    combined_path = os.path.join(
        output_dir, f"{task_name}_with_metadata_test_predictions_combined.csv"
    )
    merged_df.to_csv(combined_path, index=False)
    print(f"[SAVE] Saved combined output to: {combined_path}")
