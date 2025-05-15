import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO
import config
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import FlavaProcessor, FlavaModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from utils.logger import setup_logger

logger = setup_logger("flava_training", log_dir="runs/logs", sampled=False)
#logger = setup_logger("flava_training_log", log_dir=config.log_dir, sampled=True)


def train_flava(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    train_losses = []
    train_accuracies = []
    train_f1_scores = []  # Optional: Track F1-score for each batch

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for batch_idx, batch in enumerate(dataloader):
        if batch is None or batch['input_ids'].nelement() == 0:
            logger.warning(f"Skipping empty batch {batch_idx}")
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        metadata = batch['metadata'].to(device) if batch['metadata'] is not None else None

        logger.info(f"Batch {batch_idx} - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, pixel_values: {pixel_values.shape}, metadata: {metadata.shape if metadata is not None else 'None'}")
        # Log metadata usage for this batch
        if metadata is not None:
            logger.debug(f"Metadata included for training. Example batch metadata: {metadata[:3]}")  # Log first 3 examples
        else:
            logger.debug("Metadata not included for training.")

        try:#forward pass
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error during forward pass at batch {batch_idx}: {e}")
            continue

        if class_weights is not None:# Compute loss
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        # Accumulate metrics
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct_predictions = torch.sum(preds == labels).item()
        accuracy = correct_predictions / len(labels)

        # Optional: Calculate F1-score
        f1 = f1_score(labels.cpu(), preds.cpu(), average="weighted")

        train_losses.append(loss.item())
        train_accuracies.append(accuracy)
        train_f1_scores.append(f1)

    overall_accuracy = sum(train_accuracies) / len(train_accuracies)
    overall_f1 = sum(train_f1_scores) / len(train_f1_scores) if train_f1_scores else 0

    return total_loss / len(dataloader), train_losses, train_accuracies, overall_accuracy, overall_f1


# def train_flava(model, dataloader, optimizer, device, class_weights=None):
#     model.train()
#     total_loss = 0
#     train_losses = []
#     train_accuracies = []
#     for batch_idx, batch in enumerate(dataloader):
#         if batch is None:  # Skip empty batches
#             continue
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         pixel_values = batch['pixel_values'].to(device)
#         labels = batch['labels'].to(device)
#         metadata = batch['metadata'].to(device) if batch['metadata'] is not None else None
#         logger.info(f"Batch {batch_idx} - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, pixel_values: {pixel_values.shape}, metadata: {metadata.shape if metadata is not None else 'None'}")
        
#         # Print labels to debug the issue
#         print(f"Batch {batch_idx} labels: {labels}")
#         logging.info(f"Batch {batch_idx} labels: {labels}")

#         logits = model(
#             input_ids=input_ids, attention_mask=attention_mask,
#             pixel_values=pixel_values, metadata=metadata
#         )

#         # Print logits and labels
#         print(f"Batch {batch_idx} logits: {logits}")
#         logger.info(f"Batch {batch_idx} logits: {logits}")

#         if class_weights is not None:
#             class_weights_tensor = class_weights.to(device)
#             loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)
#         else:
#             loss = F.cross_entropy(logits, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         preds = torch.argmax(logits, dim=1)
#         correct_predictions = torch.sum(preds == labels).item()
#         accuracy = correct_predictions / len(labels)
        
#         train_losses.append(loss.item())
#         train_accuracies.append(accuracy)
    
#     return total_loss / len(dataloader), train_losses, train_accuracies