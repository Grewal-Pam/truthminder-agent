import streamlit as st
import torch
import shap
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import FlavaModel, FlavaProcessor
from models.flava_model import FlavaClassificationModel

# Load FLAVA model
@st.cache_resource
def load_model():
    flava_model = FlavaModel.from_pretrained("facebook/flava-full")
    return FlavaClassificationModel(
        flava_model=flava_model,
        num_labels=2,
        metadata_dim=3,
        include_metadata=True
    )

# Helper to load image
def load_image(url):
    try:
        response = requests.get(url, stream=True)
        image = Image.open(response.raw).convert("RGB")
        image = image.resize((224, 224))  # Resize to match FLAVA input size
        return image
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
        return None

# SHAP explanation function
def explain_prediction(model, text_input, image_input, metadata_input, training_data):
    def model_predict(metadata):
        # Convert metadata to tensor
        metadata_tensor = torch.tensor(metadata).float().to("cpu")

        # Dummy image and text handling
        text_tensor = [text_input] if text_input else None
        image_tensor = (
            torch.tensor(np.array(image))
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .float()
            if image else None
        )

        with torch.no_grad():
            output = model.predict(
                text=text_tensor,
                image=image_tensor,
                metadata=metadata_tensor.tolist(),
                processor=None,
                device="cpu"
            )
        
        # Debugging
        st.write(f"Model output: {output}")

        if not output or "probs" not in output:
            raise ValueError("Model did not return valid probabilities.")
    
        probs = np.array(output["probs"])
        st.write(f"Probabilities shape: {probs.shape}, Probabilities: {probs}")
        return probs


    # Initialize SHAP explainer
    try:
        explainer = shap.KernelExplainer(model_predict, training_metadata)
        st.write("SHAP Explainer initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SHAP explainer: {e}")
    st.write(f"Metadata input: {metadata_input}")
    st.write(f"Training data shape: {training_data.shape}")
    # Generate SHAP values
    try:
        shap_values = explainer.shap_values(metadata_input)
        st.write(f"SHAP values: {shap_values}")
        shap.summary_plot(shap_values, metadata_input)
        return plt.gcf()
    except Exception as e:
        raise RuntimeError(f"Failed to generate SHAP values: {e}")


# Streamlit UI
st.title("SHAP Explanation with FLAVA Model")

# Load the model
model = load_model()

# Input values
text_input = st.text_input("Text Input", "there is a longlegged distressed creature in my ice cream")
image_url = "https://external-preview.redd.it/Djew2GzV41YGXmfFSvwa3n3Mv91ttT1DrcAFBqcUFmA.jpg?width=320&crop=smart&auto=webp&s=5977b8d905553b61fb4a63c19d577e057bbb911d"
metadata = {
    "num_comments": 0.0,
    "score": 3,
    "upvote_ratio": 0.72
}

# Load and display the image
image = load_image(image_url)
if image:
    st.image(image, caption="Input Image")

# Prepare metadata and training data
training_metadata = np.random.rand(100, 3)  # Dummy training metadata
metadata_input = np.array([list(metadata.values())])  # Convert to array



# Generate SHAP explanation
if st.button("Run SHAP Explanation"):
    try:
        shap_fig = explain_prediction(model, text_input, image, metadata_input, training_metadata)
        st.pyplot(shap_fig)
    except Exception as e:
        st.error(f"Failed to generate SHAP explanation: {e}")

