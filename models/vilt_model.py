import config
from transformers import ViltModel
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger("vilt_model", log_dir="runs/logs", sampled=False)
#logger = setup_logger("vilt_model_log", log_dir=config.log_dir, sampled=True)


class ViltClassificationModel(nn.Module):
    def __init__(self, vilt_model, num_labels, metadata_dim=3, include_metadata=True):
        super(ViltClassificationModel, self).__init__()
        self.vilt_model = vilt_model
        self.include_metadata = include_metadata
        
        if include_metadata:
            self.classifier = nn.Linear(vilt_model.config.hidden_size + metadata_dim, num_labels)
        else:
            self.classifier = nn.Linear(vilt_model.config.hidden_size, num_labels)

        # Log model initialization details
        logger.info(f"Initialized ViltClassificationModel with include_metadata={include_metadata}")
        logger.info(f"Classifier input size: {self.classifier.in_features}, output size: {num_labels}")

    def forward(self, input_ids, attention_mask, pixel_values, metadata=None):
        logger.info(f"Forward pass started. Input shapes - input_ids: {input_ids.shape}, "
                    f"attention_mask: {attention_mask.shape}, pixel_values: {pixel_values.shape}, metadata={metadata.shape if metadata is not None else None}")

        # Log sample input values (first batch only for brevity)
        logger.debug(f"Sample input_ids (first 5): {input_ids[0, :5].tolist()}")
        logger.debug(f"Sample attention_mask (first 5): {attention_mask[0, :5].tolist()}")
        logger.debug(f"Pixel values range: min={pixel_values.min().item()}, max={pixel_values.max().item()}")

        # Step 1: ViLT Model Forward
        outputs = self.vilt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        logger.info(f"ViLT model pooler output shape: {pooled_output.shape}")
        logger.debug(f"Sample pooled_output (first 5 values): {pooled_output[0, :5].tolist()}")

        # Step 2: Combine metadata if included
        if self.include_metadata:
            if metadata is None:
                logger.warning("Metadata is None, but include_metadata=True. Defaulting to zero tensor.")
                metadata = torch.zeros((pooled_output.size(0), self.classifier.in_features - pooled_output.size(1)), 
                                       device=pooled_output.device)
            logger.info(f"Metadata shape: {metadata.shape}")
            logger.debug(f"Sample metadata values: {metadata[0, :].tolist()}")

            combined_output = torch.cat((pooled_output, metadata), dim=1)
            logger.info(f"Combined output shape (pooled_output + metadata): {combined_output.shape}")
            logger.debug(f"Sample combined_output (first 5 values): {combined_output[0, :5].tolist()}")
        else:
            combined_output = pooled_output
            logger.info(f"Using pooled output only. Shape: {combined_output.shape}")

        # Step 3: Classifier Forward
        logits = self.classifier(combined_output)
        logger.info(f"Classifier output shape (logits): {logits.shape}")
        logger.debug(f"Sample logits (first 5): {logits[0, :5].tolist()}")

        return logits
