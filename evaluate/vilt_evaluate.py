import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    recall_score,
    precision_score,
)
from utils.logger import setup_logger

logger = setup_logger("vilt_evaluation", log_dir="runs/logs", sampled=False)


# logger = setup_logger("vilt_evaluation_log")
# logger = setup_logger("vilt_evaluation_log", log_dir=config.log_dir, sampled=False)


def evaluate_vilt(model, dataloader, device, num_classes=3, average="weighted"):
    """
    Evaluate the ViLT model on the given dataloader.

    Args:
        model: The ViLT model to evaluate.
        dataloader: DataLoader for the dataset to evaluate.
        device: Torch device (e.g., 'cuda' or 'cpu').
        num_classes: Number of classes in the classification task.
        average: Averaging method for metrics (e.g., "weighted", "macro").

    Returns:
        metrics: Dictionary containing evaluation metrics.
        all_labels: List of ground truth labels.
        all_preds: List of predicted labels.
        all_probs: List of predicted probabilities for each class.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_indices = []

    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation Progress"):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            metadata = batch.get("metadata", None)
            metadata = metadata.to(device) if metadata is not None else None

            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                metadata=metadata,
            )

            # Compute loss
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            # Store predictions, labels, and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            row_indices = batch["row_index"].cpu().numpy()
            all_indices.extend(row_indices)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Average loss
    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=average)
    recall = recall_score(all_labels, all_preds, average=average)
    precision = precision_score(all_labels, all_preds, average=average)
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Classification report for per-class metrics
    class_report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "kappa": kappa,
        "class_report": class_report,  # Includes precision/recall/F1 per class
    }

    logger.info(f"Evaluation complete. Metrics: {metrics}")
    return avg_loss, metrics, all_labels, all_preds, all_probs, all_indices
