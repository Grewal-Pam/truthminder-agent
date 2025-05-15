import config
from transformers import CLIPModel
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger("clip_model" , log_dir="runs/logs", sampled=False)
#logger = setup_logger("clip_model_log", log_dir=config.log_dir, sampled=True)

class CLIPMultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, num_classes_2, num_classes_3, metadata_dim,include_metadata=True):
        super(CLIPMultiTaskClassifier, self).__init__()
        self.include_metadata = include_metadata 
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.args = (input_dim, num_classes_2, num_classes_3, metadata_dim, include_metadata)  # Store args for saving/reloading
        
        # Layers for text processing
        self.embedding = nn.Embedding(49408, input_dim)
        self.text_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        
        # Layers for metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, input_dim),
            nn.ReLU()
        )
        
        # # Classification heads
        # self.fc2 = nn.Sequential(
        #     nn.Linear(input_dim, num_classes_2),
        #     nn.Softmax(dim=1)
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(input_dim, num_classes_3),
        #     nn.Softmax(dim=1)
        # )
        # Classification heads: output raw logits (no softmax)
        self.fc2 = nn.Linear(input_dim, num_classes_2)
        self.fc3 = nn.Linear(input_dim, num_classes_3)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_dim)

    def forward(self, input_ids, attention_mask, pixel_values, metadata):
        """
        Forward pass for the multi-task classifier.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs for text (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for text input.
            pixel_values (torch.Tensor): Processed image tensor (batch_size, 3, 224, 224).
            metadata (torch.Tensor): Tabular metadata (batch_size, metadata_dim).

        Returns:
            dict: Outputs for 2-way and 3-way classification.
        """
        # Vision encoding
        vision_output = self.clip_model.get_image_features(pixel_values)
        logger.info(f"Vision output shape: {vision_output.shape}")

        # Embed input_ids
        embedded_ids = self.embedding(input_ids)
        logger.info(f"Embedded IDs shape: {embedded_ids.shape}")
        embedded_ids = embedded_ids.squeeze(1)
        text_output = self.text_fc(embedded_ids.mean(dim=1))  # Mean pooling for text features

        # Process metadata
        # metadata_output = self.metadata_fc(metadata)
        # logger.info(f"Metadata shape: {metadata.shape}, Example values: {metadata[:1]}")

        # Metadata processing branch
        if self.include_metadata:
            if metadata is None:
                logger.warning("Metadata is None, but include_metadata=True. Defaulting to zero tensor.")
                metadata = torch.zeros((vision_output.size(0), self.metadata_fc[0].in_features),
                                    device=vision_output.device)
            logger.info(f"Metadata shape: {metadata.shape}, Example values: {metadata[:1]}")
            metadata_output = self.metadata_fc(metadata)
        else:
            metadata_output = 0  # or simply omit it

        # Combine features
        combined_output = text_output + metadata_output + vision_output
        combined_output = self.feature_norm(combined_output)  # Normalize combined features
        logger.info(f"Combined output shape: {combined_output.shape}")

        # Multi-task classification
        output_2_way = self.fc2(combined_output)
        output_3_way = self.fc3(combined_output)
        logger.info(f"2-way output shape: {output_2_way.shape}")
        logger.info(f"3-way output shape: {output_3_way.shape}")

        return {"2_way": output_2_way, "3_way": output_3_way}

    def save_model(self, save_path):
        """
        Save the model's state_dict and initialization arguments.

        Args:
            save_path (str): Path to save the model.
        """
        model_state = {
            "state_dict": self.state_dict(),
            "args": self.args
        }
        torch.save(model_state, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path):
        """
        Load the model from a saved state_dict.

        Args:
            load_path (str): Path to the saved model.

        Returns:
            CLIPMultiTaskClassifier: The loaded model instance.
        """
        model_state = torch.load(load_path, map_location=torch.device('cpu'))
        model = cls(*model_state["args"])
        model.load_state_dict(model_state["state_dict"])
        logger.info(f"Model loaded from {load_path}")
        return model
