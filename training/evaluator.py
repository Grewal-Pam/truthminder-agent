import config
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import torch
from utils.logger import setup_logger
from tqdm import tqdm
from utils.plotting import Plotting

logger = setup_logger("clip_evaluation", log_dir="runs/logs", sampled=False)
#logger = setup_logger("clip_evaluation_log", log_dir=config.log_dir, sampled=True)


def evaluate_model(model, dataloader, device, labels_2=None, labels_3=None, plotter=None, mode="evaluation"):
    """
    Evaluate the model and generate metrics and plots.

    Args:
        model: The trained model.
        dataloader: DataLoader for evaluation.
        device: Torch device (CPU or CUDA).
        labels_2: List of class labels for 2-way classification.
        labels_3: List of class labels for 3-way classification.
        plotter: Plotting instance for saving graphs.
        mode: String indicating whether it's validation or test set.
    """
    logger.info(f"Starting {mode} evaluation...")
    model.eval()
    total_loss = 0
    all_labels_2, all_preds_2, all_scores_2 = [], [], []
    all_labels_3, all_preds_3, all_scores_3 = [], [], []
    row_indices = []

    model.to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{mode.capitalize()} Progress")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels2 = batch["labels_2_way"].to(device)
            labels3 = batch["labels_3_way"].to(device)
            metadata = batch.get("metadata", None)  
            if metadata is not None:
                metadata = metadata.to(device)

            outputs = model(input_ids, attention_mask, pixel_values, metadata)
            loss2 = torch.nn.functional.cross_entropy(outputs["2_way"], labels2)
            loss3 = torch.nn.functional.cross_entropy(outputs["3_way"], labels3)
            loss = loss2 + loss3

            total_loss += loss.item()

            # Append predictions and scores for 2-way classification
            all_preds_2.extend(outputs["2_way"].argmax(dim=1).cpu().tolist())
            all_scores_2.extend(torch.softmax(outputs["2_way"], dim=1)[:, 1].cpu().tolist())  # Probability of the positive class
            all_labels_2.extend(labels2.cpu().tolist())

            row_indices.extend(batch["row_index"].tolist())


            # Append predictions and scores for 3-way classification
            all_preds_3.extend(outputs["3_way"].argmax(dim=1).cpu().tolist())
            #all_scores_3.extend(torch.softmax(outputs["3_way"], dim=1).max(dim=1).values.tolist())  # Max probability
            all_scores_3.extend(torch.softmax(outputs["3_way"], dim=1).cpu().tolist())

            all_labels_3.extend(labels3.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    # Calculate metrics for 2-way classification
    metrics_2way = {
        "accuracy": accuracy_score(all_labels_2, all_preds_2),
        "precision": precision_score(all_labels_2, all_preds_2, average="weighted",  zero_division=0),
        "recall": recall_score(all_labels_2, all_preds_2, average="weighted",  zero_division=0),
        "f1": f1_score(all_labels_2, all_preds_2, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(all_labels_2, all_preds_2),
        "loss": avg_loss
    }
    
    # Calculate metrics for 3-way classification
    metrics_3way = {
        "accuracy": accuracy_score(all_labels_3, all_preds_3),
        "precision": precision_score(all_labels_3, all_preds_3, average="weighted",  zero_division=0),
        "recall": recall_score(all_labels_3, all_preds_3, average="weighted",  zero_division=0),
        "f1": f1_score(all_labels_3, all_preds_3, average="weighted",  zero_division=0),
        "kappa": cohen_kappa_score(all_labels_3, all_preds_3),
        "loss": avg_loss
    }
    
    metrics = {"2_way": metrics_2way, "3_way": metrics_3way}
    logger.info(f"{mode.capitalize()} complete. Average Loss: {avg_loss:.4f}")
    return avg_loss, metrics, all_labels_2, all_preds_2, all_scores_2, all_labels_3, all_preds_3, all_scores_3, row_indices
