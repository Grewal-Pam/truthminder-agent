import logging
import torch
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    classification_report,
)
import torch.nn.functional as F
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("flava_evaluate", log_dir="runs/logs", sampled=False)
# logger = setup_logger("flava_evaluate_log", log_dir=config.log_dir, sampled=False)


# Evaluation function
def evaluate_flava(model, dataloader, device, task_name="2-way"):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []
    val_losses = []
    val_accuracies = []
    all_probs = []
    all_indices = []
    logging.info("Starting evaluation loop for {task_name} classification...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # if batch is None or batch['labels'].nelement() == 0:
            if (
                batch is None
                or batch.get("labels") is None
                or len(batch["labels"]) == 0
            ):
                logger.warning(f"Skipping empty batch {batch_idx} during evaluation.")
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            metadata = (
                batch["metadata"].to(device) if batch["metadata"] is not None else None
            )
            ids = (
                batch["row_index"].to(device)
                if isinstance(batch["row_index"], torch.Tensor)
                else batch["row_index"]
            )

            if labels.nelement() == 0:  # Check if labels are empty
                logger.error(f"Batch {batch_idx} has empty labels. Skipping...")
                continue

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                metadata=metadata,
            )
            # Debugging statement
            print(
                f"Logits shape: {logits.shape}, Expected shape: (batch_size, num_classes)"
            )
            logger.info(
                f"Logits shape: {logits.shape}, Expected shape: (batch_size, num_classes)"
            )

            # Add probabilities for ROC/PR curves
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            # Debug: Print shapes of logits and labels
            print(
                f"Eval batch {batch_idx}: logits shape: {logits.shape}, labels shape: {labels.shape}"
            )
            logger.info(
                f"Eval batch {batch_idx}: logits shape: {logits.shape}, labels shape: {labels.shape}"
            )

            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            all_indices.extend(
                ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids
            )

            accuracy = torch.sum(preds == labels).item() / len(labels)
            val_losses.append(loss.item())
            val_accuracies.append(accuracy)

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Evaluation Loss: {loss.item()}")

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)  # Concatenate probabilities

    all_probs = np.vstack(all_probs)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted")
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    # Determine number of classes dynamically from predictions (safe fallback)
    num_classes = len(np.unique(all_labels))

    # Create per-class report
    class_report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "loss": float(total_loss / len(dataloader)),
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "class_report": class_report,
    }
    print(f"→ Number of predicted classes: {len(np.unique(all_preds))}")
    print(f"→ Predicted classes: {np.unique(all_preds)}")

    return metrics, (
        all_labels,
        all_preds,
        all_probs,
        val_losses,
        val_accuracies,
        all_indices,
    )
