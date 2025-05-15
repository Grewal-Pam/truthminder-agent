from models.flava_model import FlavaClassificationModel
from transformers import FlavaModel
import pandas as pd

# Load the pretrained FLAVA model
flava_model = FlavaModel.from_pretrained("facebook/flava-full")

# Initialize the classification model
model = FlavaClassificationModel(flava_model, num_labels=2)

# Check if the 'predict' method exists
print("Does 'predict' method exist?", hasattr(model, "predict"))  # Should print True
