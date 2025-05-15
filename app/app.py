import streamlit as st
import os
import json
import torch
from models.clip_model import CLIPMultiTaskClassifier
from models.flava_model import FlavaClassificationModel
from models.vilt_model import ViltClassificationModel
from transformers import FlavaModel, FlavaProcessor
from PIL import Image
import requests
from io import BytesIO
import logging
import shap
import matplotlib.pyplot as plt
import pickle
import logging
from utils.logger import setup_logger
logger = setup_logger("shap_debug_log")

# Set up logging
#logging.basicConfig(level=logging.INFO)

# Define the load_image function
def load_image(url):
    """
    Load an image from a URL and return a PIL Image object.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))  # Resize to match model input requirements
        return img
    except Exception as e:
        logging.error(f"Failed to load image from URL {url}: {e}")
        return None

# SHAP Explanation Function
def explain_prediction(model, metadata_input, training_data):
    logger.info("Starting SHAP explanation.")
    try:
        def model_predict(metadata):
            logger.info(f"Predicting with metadata: {metadata}")
            return model(metadata).detach().numpy()

        explainer = shap.Explainer(model_predict, training_data)
        logger.info("SHAP explainer initialized.")
        
        shap_values = explainer(metadata_input)
        logger.info(f"SHAP values calculated: {shap_values}")

        shap_fig = shap.summary_plot(shap_values, metadata_input, show=False)
        logger.info("SHAP summary plot created.")
        return shap_fig
    except Exception as e:
        logger.error(f"Error in explain_prediction: {e}")
        raise

# Set page configuration
st.set_page_config(page_title="Disinformation Detection using a Multimodal Dataset", layout="wide")

# Sidebar for model selection
st.sidebar.title("Model Options")
model_type = st.sidebar.selectbox("Select Model", ["CLIP", "FLAVA", "ViLT"])
classification_type = st.sidebar.radio("Classification Type", ["2-way_classification", "3-way_classification"])

# Main title
st.title("Multimodal Disinformation Detection")
st.write("Upload text, image, or metadata for prediction.")

# Text input
uploaded_text = st.text_area("Enter Text for Prediction", "")

# Image input
st.write("Provide an Image:")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or Paste an Image URL", "")

# Metadata input
st.write("Provide Metadata (Optional):")
num_comments = st.number_input("Number of Comments", min_value=0, value=0)
score = st.number_input("Score", min_value=0, value=0)
upvote_ratio = st.slider("Upvote Ratio", min_value=0.0, max_value=1.0, value=0.5)

# Metadata Preparation
metadata = {
    "num_comments": num_comments,
    "score": score,
    "upvote_ratio": upvote_ratio
}

# Load training metadata for SHAP
metadata_save_path = "app/assets/FLAVA/training_metadata.pkl"
if os.path.exists(metadata_save_path):
    with open(metadata_save_path, "rb") as f:
        training_metadata = pickle.load(f)
else:
    training_metadata = None
    st.warning("Training metadata not found. SHAP explanations will not work.")

# Load model
@st.cache_resource
def load_model(model_type, classification_type):
    num_labels = 2 if classification_type == "2-way_classification" else 3

    if model_type == "CLIP":
        return CLIPMultiTaskClassifier(num_labels=num_labels)
    elif model_type == "FLAVA":
        flava_model = FlavaModel.from_pretrained("facebook/flava-full")
        model = FlavaClassificationModel(
            flava_model=flava_model,
            num_labels=num_labels,
            metadata_dim=3,
            include_metadata=True
        )
        return model
    elif model_type == "ViLT":
        return ViltClassificationModel(num_labels=num_labels)

if st.sidebar.button("Load Model"):
    st.session_state.model = load_model(model_type, classification_type)
    st.success(f"{model_type} {classification_type} model loaded.")

# Image Handling
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image")
elif image_url:
    image = load_image(image_url)
    if image:
        st.image(image, caption="Image from URL")
    else:
        st.error("Failed to load image from the provided URL.")

# Initialize the processor
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

# Prediction
if st.button("Predict"):
    if not uploaded_text and not image:
        st.warning("Please provide at least text or an image for prediction.")
    elif "model" not in st.session_state:
        st.error("Model not loaded. Please load a model first.")
    else:
        st.write("### Input Summary")
        if uploaded_text:
            st.write(f"**Text**: {uploaded_text}")
        if image:
            st.write("**Image Provided**")
        st.write("**Metadata**:")
        st.json(metadata)

        # Predict using the loaded model
        prediction = st.session_state.model.predict(
            text=uploaded_text,
            image=image,
            metadata=list(metadata.values()),
            processor=processor,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        st.write("### Prediction")
        st.write(f"**Prediction**: {prediction['label']}")
        st.write(f"**Confidence**: {prediction['confidence']:.2f}%")
        
        # SHAP Explanation
        if st.checkbox("Explain Prediction (SHAP)", key="shap_checkbox"):
            logging.info("SHAP checkbox is clicked.")
            if "shap_checkbox" in st.session_state and st.session_state.shap_checkbox:
                logging.info("SHAP checkbox is active.")
                try:
                    metadata_tensor = torch.tensor(list(metadata.values())).unsqueeze(0).to("cpu")
                    logging.info(f"Metadata tensor: {metadata_tensor}")
                    if training_metadata is not None:
                        training_metadata_tensor = torch.tensor(training_metadata).to("cpu")
                        logging.info(f"Training metadata tensor: {training_metadata_tensor.shape}")
                        
                        # Call the SHAP explanation function
                        shap_fig = explain_prediction(st.session_state.model, metadata_tensor, training_metadata_tensor)
                        
                        # Display the SHAP summary plot
                        st.pyplot(shap_fig)
                    else:
                        logging.warning("Training metadata is missing.")
                        st.error("Training metadata is not available. SHAP explanations cannot be generated.")
                except Exception as e:
                    logging.error(f"SHAP explanation failed: {e}")
                    st.error("Failed to generate SHAP explanation. Check logs for details.")




# Metrics and Graphs
if st.checkbox("Show Metrics and Graphs"):
    graph_folder = f"app/assets/{model_type}/graphs/"
    metrics_folder = f"app/assets/{model_type}/metrics/"
    classification_filter = classification_type.replace("_classification", "")

    if os.path.exists(graph_folder):
        graph_files = [
            file for file in os.listdir(graph_folder)
            if file.endswith(".png") and classification_filter in file
        ]
        if graph_files:
            st.subheader(f"Graphs for {classification_type}")
            for graph_file in graph_files:
                graph_path = os.path.join(graph_folder, graph_file)
                st.image(graph_path, caption=graph_file)
        else:
            st.error(f"No graphs found for {classification_type}.")
    else:
        st.error(f"Graph folder not found: {graph_folder}")

    if os.path.exists(metrics_folder):
        metrics_files = [
            file for file in os.listdir(metrics_folder)
            if file.endswith(".json") and classification_filter in file
        ]
        if metrics_files:
            st.subheader(f"Metrics for {classification_type}")
            for metrics_file in metrics_files:
                metrics_path = os.path.join(metrics_folder, metrics_file)
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                st.json({metrics_file: metrics})
        else:
            st.error(f"No metrics found for {classification_type}.")
    else:
        st.error(f"Metrics folder not found: {metrics_folder}")


