import os
import logging
import pandas as pd
from local_datasets.dataset import load_dataset, load_images, preprocess_image

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_dataset_with_images():
    # Step 1: Ensure the dataset file exists
    dataset_path = 'data/filtered_data.tsv'
    assert os.path.exists(dataset_path), f"Dataset file {dataset_path} is missing."

    # Step 2: Load the dataset
    train_df, val_df, test_df = load_dataset(file_path=dataset_path)
    assert len(train_df) > 0, "Training set should not be empty."
    assert len(val_df) > 0, "Validation set should not be empty."
    assert len(test_df) > 0, "Test set should not be empty."
    
    logger.info(f"Loaded dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Step 3: Apply load_images to preprocess image URLs in the training set
    processed_train_df = load_images(train_df)
    logger.info(f"Processed training set: {len(processed_train_df)} rows with valid images")

    # Step 4: Assert results
    assert 'image' in processed_train_df.columns, "'image' column missing after processing."
    assert len(processed_train_df) > 0, "No valid images were loaded in the training set."
    assert processed_train_df['image'].notnull().all(), "Some images failed to load."

    # Log some sample outputs for debugging
    logger.info(f"Sample processed rows:\n{processed_train_df.head()}")



def test_preprocess_image():
    valid_url = 'https://external-preview.redd.it/WylDbZrnbvZdBpgfa3ntxYf17CBHndiJWHylVm2j_nY.jpg?width=320&crop=smart&auto=webp&s=449659a10792de4d55c2f27d2176fdc8bc66e72a'  # Example of a valid image URL
    invalid_url = 'https://invalid-url.com/invalid.jpg'

    # Test valid image URL
    img = preprocess_image(valid_url)
    assert img is not None, "Image loading failed for valid URL"
    assert img.size == (224, 224), "Image was not resized correctly to (224, 224)"

    # Test invalid image URL
    img = preprocess_image(invalid_url)
    assert img is None, "Invalid URL should return None"


def test_load_images():
    data = {
        'image_url': [
            'https://external-preview.redd.it/WylDbZrnbvZdBpgfa3ntxYf17CBHndiJWHylVm2j_nY.jpg?width=320&crop=smart&auto=webp&s=449659a10792de4d55c2f27d2176fdc8bc66e72a'
            #'https://invalid-url.com/invalid.jpg'  # Invalid image URL
        ]
    }
    df = pd.DataFrame(data)

    # Apply load_images
    processed_df = load_images(df)

    # Check results
    assert 'image' in processed_df.columns, "'image' column is missing after processing"
    assert len(processed_df) == 1, "Invalid image URL should be filtered out"
    assert processed_df['image'].iloc[0].size == (224, 224), "Valid image was not resized correctly"
