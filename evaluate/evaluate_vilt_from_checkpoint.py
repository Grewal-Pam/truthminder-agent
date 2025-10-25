import os
import torch
import pandas as pd
import numpy as np
from transformers import ViltProcessor
from transformers import ViltModel
from models.vilt_model import ViltClassificationModel
from dataloaders.vilt_dataloader import get_vilt_dataloader
from evaluate.vilt_evaluate import evaluate_vilt  # Assuming correct import path
import matplotlib.pyplot as plt


def analyze_metadata_influence(
    model,
    test_df,
    processor,
    label_type,
    metadata_columns,
    experiment_name,
    device="cpu",
    plot=False,
):

    print("🔍 Estimating metadata feature influence (post-hoc ablation)...")
    feature_names = metadata_columns
    influence_scores = []

    test_loader, _ = get_vilt_dataloader(
        test_df,
        processor,
        label_type=label_type,
        batch_size=1,
        shuffle=False,
        metadata_columns=metadata_columns,
    )

    model.eval()

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        metadata = batch["metadata"].to(device)

        with torch.no_grad():
            original_logits = model(input_ids, attention_mask, pixel_values, metadata)
            original_probs = torch.softmax(original_logits, dim=-1)

        row_influence = []
        for i in range(metadata.shape[1]):
            modified_metadata = metadata.clone()
            modified_metadata[0, i] = 0.0
            with torch.no_grad():
                modified_logits = model(
                    input_ids, attention_mask, pixel_values, modified_metadata
                )
                modified_probs = torch.softmax(modified_logits, dim=-1)
            delta = torch.abs(original_probs - modified_probs).sum().item()
            row_influence.append(delta)

        influence_scores.append(row_influence)

    influence_array = np.array(influence_scores)
    avg_influence = influence_array.mean(axis=0)

    print("📊 Average Influence per Metadata Feature:")
    for fname, score in zip(feature_names, avg_influence):
        print(f"   - {fname}: {score:.4f}")

    output_dir = os.path.join("runs", experiment_name, "metrics")
    os.makedirs(output_dir, exist_ok=True)

    # Save average influence
    avg_df = pd.DataFrame([avg_influence], columns=feature_names)
    avg_filename = f"{label_type.replace('_label','')}_metadata_influence.csv"
    avg_df.to_csv(os.path.join(output_dir, avg_filename), index=False)

    # Save per-row influence
    row_df = pd.DataFrame(
        influence_array, columns=[f"inf_{col}" for col in feature_names]
    )
    row_filename = f"{label_type.replace('_label','')}_metadata_influence_per_row.csv"
    row_path = os.path.join(output_dir, row_filename)
    row_df.to_csv(row_path, index=False)
    print(f"📄 Saved per-row influence to: {row_filename}")

    # Optional plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(feature_names, avg_influence, color="skyblue")
        plt.ylabel("Influence Score (Δ Softmax)")
        plt.title(f"Metadata Influence - {label_type.replace('_label','').upper()}")
        plt.tight_layout()
        plot_filename = f"{label_type.replace('_label','')}_metadata_influence.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()
        print(f"📊 Saved plot to: {plot_filename}")

    return row_path  # 👈 return path to per-row file for merging


# --- CONFIGURATION ---
EXPERIMENT_NAME = "VILT/2025-04-06_16-48-46_GOOD"  # "VILT/2025-03-30_10-46-27"  # adjust to your folder
test_batch_file = "data/test_batch_1.tsv"
BATCH_SIZE = 16
DEVICE = "cpu"
LABEL_TYPES = {"2_way_label": 2, "3_way_label": 3}
METADATA_COLUMNS = ["num_comments", "score", "upvote_ratio"]
# ----------------------

print("📂 Loading test data...")
test_df = pd.read_csv(test_batch_file, sep="\t").sample(n=20, random_state=42)

# Load processor
print("🔄 Loading ViLT processor...")
# Load base ViLT model
vilt_base = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Loop through both label types and metadata configs
for label_type, num_classes in LABEL_TYPES.items():
    for include_metadata in [True, False]:
        metadata_tag = "with_metadata" if include_metadata else "without_metadata"
        task_name = f"{label_type.replace('_label', '')}_classification"

        print(f"\n🚀 Evaluating {task_name} ({metadata_tag})")

        # Get dataloader
        print("📦 Preparing test dataloader...")
        test_loader, _ = get_vilt_dataloader(
            test_df,
            processor,
            label_type=label_type,
            batch_size=BATCH_SIZE,
            shuffle=False,
            metadata_columns=METADATA_COLUMNS if include_metadata else None,
        )

        # Load model checkpoint
        # checkpoint_name = f"vilt_final_model_{task_name.replace(' ', '-')}_{metadata_tag}.pth"
        label_base = label_type.replace("_label", "").replace("_", "-")
        task_name = f"{label_base}_classification"
        checkpoint_name = f"vilt_final_model_{task_name}_{metadata_tag}.pth"

        checkpoint_path = os.path.join(
            "runs", EXPERIMENT_NAME, "models", checkpoint_name
        )
        print(f"📥 Loading model from: {checkpoint_path}")

        model = ViltClassificationModel(
            vilt_model=vilt_base,
            num_labels=num_classes,
            metadata_dim=len(METADATA_COLUMNS) if include_metadata else 0,
            include_metadata=include_metadata,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # Run evaluation
        print("🧪 Running model evaluation...")
        loss, metrics, labels, preds, probs, row_indices = evaluate_vilt(
            model, test_loader, DEVICE
        )

        if include_metadata:
            influence_path = analyze_metadata_influence(
                model=model,
                test_df=test_df,
                processor=processor,
                label_type=label_type,
                metadata_columns=METADATA_COLUMNS,
                experiment_name=EXPERIMENT_NAME,
                device=DEVICE,
                plot=True,
            )

        # Trace original rows
        print("📌 Collecting prediction results...")
        used_test_df = test_df.iloc[row_indices].reset_index(drop=True)
        results_df = pd.DataFrame({"true_label": labels, "predicted_label": preds})
        probs_df = pd.DataFrame(
            np.array(probs),
            columns=[f"prob_class_{i}" for i in range(np.array(probs).shape[1])],
        )

        final_df = pd.concat(
            [used_test_df.reset_index(drop=True), results_df, probs_df], axis=1
        )

        # Save results
        output_dir = os.path.join("runs", EXPERIMENT_NAME, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = f"{task_name}_{metadata_tag}_test_predictions_detailed.csv"
        final_path = os.path.join(output_dir, csv_filename)
        final_df.to_csv(final_path, index=False)

        # 🔁 Merge per-row influence scores
        if include_metadata:
            influence_df = pd.read_csv(influence_path)
            final_combined_df = pd.concat(
                [final_df.reset_index(drop=True), influence_df.reset_index(drop=True)],
                axis=1,
            )
            combined_filename = (
                f"{task_name}_{metadata_tag}_test_predictions_combined.csv"
            )
            combined_path = os.path.join(output_dir, combined_filename)
            final_combined_df.to_csv(combined_path, index=False)
            print(f"📄 Saved combined predictions + influence to: {combined_path}")

        # print(f"✅ Saved predictions to: {final_path}")
        print(
            f"📊 Evaluation metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
        )
