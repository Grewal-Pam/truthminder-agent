import torch.nn.functional as F
import torch
from utils.logger import setup_logger
from tqdm import tqdm
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)

logger = setup_logger("clip_training", log_dir="runs/logs", sampled=False)
# logger = setup_logger("clip_training_log", log_dir=config.log_dir, sampled=True)


def train_model(
    model,
    dataloader,
    optimizer,
    device,
    clip_grad=False,
    save_dir="results",
    model_name="model_weights.pth",
):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.path.join(save_dir, model_name)
    logger.info("Starting training...")
    model.train()
    epoch_loss = 0.0
    all_preds_2, all_labels_2 = [], []
    all_preds_3, all_labels_3 = [], []

    # Validate Pixel Values
    first_batch = next(iter(dataloader))
    logger.info(
        f"Example pixel values (first 5): {first_batch['pixel_values'][0].view(-1)[:5].tolist()}"
    )
    logger.info(f"Pixel values shape: {first_batch['pixel_values'][0].shape}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Progress")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels2 = batch["labels_2_way"].to(device)
        labels3 = batch["labels_3_way"].to(device)
        metadata = (
            batch["metadata"].to(device)
            if "metadata" in batch and batch["metadata"].numel() > 0
            else None
        )

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask, pixel_values, metadata)

        # Compute loss
        loss2 = F.cross_entropy(outputs["2_way"], labels2)
        loss3 = F.cross_entropy(outputs["3_way"], labels3)
        loss = loss2 + loss3

        # Check for NaN
        if torch.isnan(loss):
            logger.error(f"NaN detected in loss at batch {batch_idx}")
            raise ValueError(f"NaN detected in loss at batch {batch_idx}")

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()
        # total_loss += loss.item()
        # Track predictions
        all_preds_2.extend(outputs["2_way"].argmax(dim=1).tolist())
        all_labels_2.extend(labels2.tolist())
        all_preds_3.extend(outputs["3_way"].argmax(dim=1).tolist())
        all_labels_3.extend(labels3.tolist())

        logger.info(f"[Batch {batch_idx + 1}] Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    # Metrics for 2-way classification
    metrics_2 = {
        "accuracy": accuracy_score(all_labels_2, all_preds_2),
        "precision": precision_score(all_labels_2, all_preds_2, average="weighted"),
        "recall": recall_score(all_labels_2, all_preds_2, average="weighted"),
        "f1": f1_score(all_labels_2, all_preds_2, average="weighted"),
        "kappa": cohen_kappa_score(all_labels_2, all_preds_2),
    }

    # Metrics for 3-way classification
    metrics_3 = {
        "accuracy": accuracy_score(all_labels_3, all_preds_3),
        "precision": precision_score(all_labels_3, all_preds_3, average="weighted"),
        "recall": recall_score(all_labels_3, all_preds_3, average="weighted"),
        "f1": f1_score(all_labels_3, all_preds_3, average="weighted"),
        "kappa": cohen_kappa_score(all_labels_3, all_preds_3),
    }

    logger.info(f"Average Epoch Loss: {avg_loss:.4f}")
    logger.info(
        f"[2-way] Accuracy: {metrics_2['accuracy']:.4f}, F1: {metrics_2['f1']:.4f}"
    )
    logger.info(
        f"[3-way] Accuracy: {metrics_3['accuracy']:.4f}, F1: {metrics_3['f1']:.4f}"
    )
    return avg_loss, metrics_2, metrics_3

    # # Save model
    # torch.save(model.state_dict(), save_path)
    # logger.info(f"Model saved to {save_path}")
    # model_weights = torch.load("results/model_weights.pth")
    # logger.info(model_weights.keys())  # Shows all saved parameters

    # # Load and verify model
    # loaded_model = type(model)(*model.args)  # Recreate the model with same arguments
    # loaded_model.load_state_dict(torch.load(save_path))
    # loaded_model.to(device)
    # loaded_model.eval()

    # # Test inference on a sample input
    # with torch.no_grad():
    #     test_input = {
    #         'input_ids': first_batch['input_ids'].to(device),
    #         'attention_mask': first_batch['attention_mask'].to(device),
    #         'pixel_values': first_batch['pixel_values'].to(device),
    #         'metadata': first_batch['metadata'].to(device)
    #     }
    #     test_output = loaded_model(
    #         test_input['input_ids'],
    #         test_input['attention_mask'],
    #         test_input['pixel_values'],
    #         test_input['metadata']
    #     )
    # logger.info(f"Test output (2-way example): {test_output['2_way'][0]}")
    # logger.info(f"Test output (3-way example): {test_output['3_way'][0]}")

    # return total_loss / num_epochs
