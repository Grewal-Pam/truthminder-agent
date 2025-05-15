import config
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.logger import setup_logger

logger = setup_logger("vilt_training", log_dir="runs/logs", sampled=False)


def train_vilt(model, dataloader, optimizer, device, class_weights=None):
    """
    Train the ViLT model.
    """
    model.train()
    total_loss = 0
    train_losses = []
    train_accuracies = []

    logger.info("Starting training loop...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training Progress")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        metadata = batch["metadata"].to(device) if batch["metadata"] is not None else None

        # Forward pass
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            metadata=metadata,
        )

        # Compute loss
        if class_weights is not None:
            class_weights_tensor = class_weights.to(device)
            loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)
        else:
            loss = F.cross_entropy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy calculation
        preds = torch.argmax(logits, dim=1)
        correct_predictions = torch.sum(preds == labels).item()
        accuracy = correct_predictions / len(labels)

        train_losses.append(loss.item())
        train_accuracies.append(accuracy)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = sum(train_accuracies) / len(train_accuracies)

    logger.info(f"Training complete. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
    return avg_loss, train_losses, avg_accuracy
