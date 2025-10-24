import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger("flava_model", log_dir="runs/logs", sampled=False)
# logger = setup_logger("flava_model_log", log_dir=config.log_dir, sampled=True)


class FlavaClassificationModel(nn.Module):
    def __init__(self, flava_model, num_labels, metadata_dim=3, include_metadata=True):
        super(FlavaClassificationModel, self).__init__()
        self.flava_model = flava_model
        self.include_metadata = include_metadata
        self.metadata_dim = metadata_dim
        self.num_labels = num_labels

        if include_metadata:
            self.classifier = nn.Linear(
                flava_model.config.hidden_size + metadata_dim, num_labels
            )
        else:
            self.classifier = nn.Linear(flava_model.config.hidden_size, num_labels)
        logger.info(f"Number of labels used here={num_labels}")

    def update_num_labels(self, num_labels):
        """Update the classifier for a new number of labels."""
        self.num_labels = num_labels
        if self.include_metadata:
            self.classifier = nn.Linear(
                self.flava_model.config.hidden_size + self.metadata_dim, num_labels
            )
        else:
            self.classifier = nn.Linear(self.flava_model.config.hidden_size, num_labels)
        logger.info(f"Updated number of labels to: {num_labels}")

    def forward(self, input_ids, attention_mask, pixel_values, metadata=None):
        logger.info(
            f"Inside forward: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, pixel_values={pixel_values.shape}, metadata={metadata.shape if metadata is not None else None}"
        )

        outputs = self.flava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        pooled_output = outputs.multimodal_output.pooler_output

        if self.include_metadata and metadata is not None:
            combined_output = torch.cat((pooled_output, metadata), dim=1)
        else:
            combined_output = pooled_output

        logits = self.classifier(combined_output)
        return logits

    def predict_old(
        self, text=None, image=None, metadata=None, processor=None, device="cpu"
    ):
        """Perform inference using text, image, and metadata inputs."""
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            # Preprocess inputs
            inputs = {}
            if text:
                text_inputs = processor(
                    text=text, return_tensors="pt", padding=True, truncation=True
                )
                inputs["input_ids"] = text_inputs["input_ids"].to(device)
                inputs["attention_mask"] = text_inputs["attention_mask"].to(device)

            if image:
                image_tensor = processor(images=image, return_tensors="pt")[
                    "pixel_values"
                ].to(device)
                inputs["pixel_values"] = image_tensor

            if metadata:
                metadata_tensor = (
                    torch.tensor(metadata, dtype=torch.float32).unsqueeze(0).to(device)
                )
            else:
                metadata_tensor = None

            # Forward pass
            logits = self.forward(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                metadata=metadata_tensor,
            )

            # Postprocess outputs
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()

            return {
                "label": predicted_label,
                "confidence": probabilities.max().item() * 100,
            }

    def predict(
        self, text=None, image=None, metadata=None, processor=None, device="cpu"
    ):
        """Perform inference using text, image, and optional metadata inputs."""
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            # Preprocess inputs
            inputs = {}
            if text:
                text_inputs = processor(
                    text=text, return_tensors="pt", padding=True, truncation=True
                )
                inputs["input_ids"] = text_inputs["input_ids"].to(device)
                inputs["attention_mask"] = text_inputs["attention_mask"].to(device)

            if image:
                image_tensor = processor(images=image, return_tensors="pt")[
                    "pixel_values"
                ].to(device)
                inputs["pixel_values"] = image_tensor

            # Handle metadata
            if self.include_metadata:  # Check if the model was trained with metadata
                if metadata:
                    metadata_tensor = (
                        torch.tensor(metadata, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                else:
                    raise ValueError(
                        "Metadata is required for this model but was not provided."
                    )
            else:
                metadata_tensor = None

            # Forward pass
            logits = self.forward(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                metadata=metadata_tensor,
            )

            # Postprocess outputs
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()

            return {
                "label": predicted_label,
                "confidence": probabilities.max().item() * 100,
            }
