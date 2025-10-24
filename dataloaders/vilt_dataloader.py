import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.logger import setup_logger
from PIL import Image
import requests
from io import BytesIO
import pytesseract

logger = setup_logger("vilt_dataloader", log_dir="runs/logs", sampled=False)
# logger = setup_logger("vilt_dataloader_log", log_dir=config.log_dir, sampled=True)


class ViLTDataset(Dataset):
    def __init__(
        self, dataframe, processor, label_type="2_way_label", metadata_columns=None
    ):
        self.dataframe = dataframe
        self.processor = processor
        self.label_type = (
            label_type  # This will be used to select which label column to use
        )
        self.metadata_columns = metadata_columns or []

        # Ensure metadata is numeric and handle missing values
        for col in self.metadata_columns:
            self.dataframe[col] = pd.to_numeric(
                self.dataframe[col], errors="coerce"
            ).fillna(0)

        # Calculate class weights for the specified label_type
        label_counts = self.dataframe[label_type].value_counts().sort_index()
        if len(label_counts) > 1:
            class_weights = torch.tensor(
                label_counts.sum() / (len(label_counts) * label_counts),
                dtype=torch.float32,
            )
            self.class_weights = class_weights / class_weights.sum()
        else:
            self.class_weights = None

        logger.info(
            f"Initialized ViLTDataset with {len(self.dataframe)} samples and class weights: {self.class_weights}"
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["clean_title"]

        # Instead of looking for 'image', use 'image_url' to load and process the image.
        image_url = row.get("image_url", None)
        if image_url is None or pd.isna(image_url):
            logger.error(f"Missing image URL in row {idx}: {row}")
            # raise ValueError(f"Missing image URL in row {idx}")
            return None  # Skip this row

        # Process the image on the fly using your preprocess_image function.
        image = preprocess_image(image_url)
        if image is None:
            logger.error(f"Could not process image at URL: {image} in row {idx}")
            # raise ValueError(f"Could not process image at URL: {image} in row {idx}")
            return None

        # Get the label from the specified label_type column.
        label = torch.tensor(row[self.label_type], dtype=torch.long)

        # Handle metadata: if metadata_columns are provided, extract them.
        if self.metadata_columns:
            raw_metadata = row[self.metadata_columns]
            logger.debug(f"Row {idx} raw metadata: {raw_metadata}")
            # metadata_values = row[self.metadata_columns].values.astype(float)
            metadata_values = raw_metadata.astype(float).fillna(0).values
            metadata = torch.tensor(metadata_values, dtype=torch.float32)
        else:
            metadata = torch.empty(0, dtype=torch.float32)

        # Process inputs with the processor.
        # The processor is expected to compute pixel values from the image.
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40,
        )
        inputs["row_index"] = torch.tensor(idx)  # Track original row index

        # Remove extra batch dimension.
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        # Add metadata and label to the inputs.
        inputs["metadata"] = metadata
        # Here, you can store the label under a generic key;
        # for ViLT, you may have separate keys for different label types.
        # For example, if label_type is '2_way_label', you can use:
        if self.label_type == "2_way_label":
            inputs["labels"] = label
        elif self.label_type == "3_way_label":
            inputs["labels"] = label
        else:
            # Otherwise, you can store under a default key.
            inputs["labels"] = label

        logger.debug(
            f"Processed sample {idx}: label shape {inputs['labels'].shape}, metadata shape {inputs['metadata'].shape}"
        )
        return inputs


def is_placeholder_image(img):
    """
    Detect if an image is likely a placeholder based on simple heuristics.
    This function checks:
      - If the image is completely black or white.
      - If the image has very low variance (almost a uniform color).
      - Optionally, if OCR detects known placeholder phrases.
    """
    try:
        # Convert image to grayscale
        grayscale = img.convert("L")
        # Get pixel extrema (min and max)
        extrema = grayscale.getextrema()
        if extrema == (0, 0):
            logger.warning("Image is completely black.")
            return True
        if extrema == (255, 255):
            logger.warning("Image is completely white.")
            return True

        # Check for low variance: if a single value dominates the histogram.
        histogram = grayscale.histogram()
        total_pixels = sum(histogram)
        max_pixel_count = max(histogram)
        if total_pixels > 0 and (max_pixel_count / total_pixels) > 0.98:
            # logger.warning("Image has very low variance; likely a placeholder.")
            return True

        # Optional: OCR-based detection
        try:
            text = pytesseract.image_to_string(img)
            placeholder_keywords = [
                "not found",
                "unavailable",
                "error",
                "placeholder",
                "no longer available",
            ]
            for keyword in placeholder_keywords:
                if keyword.lower() in text.lower():
                    # logger.warning(f"Detected placeholder keyword '{keyword}' in image.")
                    return True
        except Exception as e:
            logger.warning(f"OCR processing failed: {e}")

    except Exception:
        # logger.error(f"Error during placeholder detection: {e}")
        # If something goes wrong, better treat it as a placeholder to skip it.
        return True

    return False


def preprocess_image(url, size=(224, 224)):
    """
    Download and preprocess an image from a URL.
    Returns a resized PIL image in RGB mode if successful, otherwise None.
    """
    session = requests.Session()
    # Update headers to mimic a real browser
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Referer": url,  # Simulate a referer header with the URL itself
        }
    )
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Check if the image is empty
        if img.size == (0, 0):
            # logger.error(f"Empty image at URL: {url}")
            return None

        # Check if the image is a placeholder
        if is_placeholder_image(img):
            # logger.warning(f"Placeholder image detected at URL: {url}")
            return None

        # Resize the image to the specified size
        processed_img = img.resize(size)
        return processed_img

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            logger.warning(f"Image not found (404): {url}")
        else:
            logger.error(f"HTTP error for image {url}: {http_err}")
    except requests.RequestException as req_err:
        logger.error(f"Request error for image {url}: {req_err}")
    except Exception as e:
        logger.error(f"Error processing image {url}: {e}")

    return None


def collate_fn(batch):
    # Filter out None values before processing
    batch = [item for item in batch if item is not None and item["labels"].numel() > 0]
    # If nothing remains, return None (skip batch)
    if len(batch) == 0:
        return None
    row_indices = torch.stack([item["row_index"] for item in batch])

    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    pixel_values = torch.cat(
        [
            (
                item["pixel_values"].unsqueeze(0)
                if item["pixel_values"].dim() == 3
                else item["pixel_values"]
            )
            for item in batch
        ],
        dim=0,
    )
    labels = torch.cat(
        [
            item["labels"].unsqueeze(0) if item["labels"].dim() == 0 else item["labels"]
            for item in batch
        ],
        dim=0,
    )

    metadata = (
        torch.stack([item["metadata"] for item in batch], dim=0)
        if "metadata" in batch[0]
        else torch.tensor([], dtype=torch.float32)
    )

    logger.info(
        f"Collated batch, input_ids shape: {input_ids.shape}, labels shape: {labels.shape}"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels.squeeze(),
        "metadata": metadata if metadata.nelement() > 0 else None,
        "row_index": row_indices,
    }


def get_vilt_dataloader(
    df, processor, label_type, batch_size=32, shuffle=True, metadata_columns=None
):
    dataset = ViLTDataset(df, processor, label_type, metadata_columns)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader, dataset.class_weights
