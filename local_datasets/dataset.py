import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import requests
from io import BytesIO
import logging
import pickle
import os
from local_datasets.pre_processing import compute_pixel_values
from torch.utils.data import DataLoader
from utils.helpers import load_images


# Constants
SEED = 42
DATA_DIR = 'data'
FEATURE_DIR = 'features'


def load_dataset(file_path="data/filtered_data.tsv", test_size=0.2, val_size=0.25, seed=SEED):
    df = pd.read_csv(file_path, sep='\t')
     # Compute pixel values
    df = compute_pixel_values(df, image_column=image_url)
    #logger.info(f"Loaded dataset with {len(df)} rows.")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed)
    return train_df, val_df, test_df


"""
 def handle_missing_values(df, columns):
    return df.dropna(subset=columns)

def normalize_metadata(df, metadata_columns, scaler_path=None):
    scaler = StandardScaler()
    df[metadata_columns] = scaler.fit_transform(df[metadata_columns])
    if scaler_path:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    return df
 """
def preprocess_image(url, size=(224, 224)):
    """
    Loads an image from a URL, resizes it, and converts it to RGB.
    If loading fails, returns None.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()  # Check for HTTP request errors
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img.resize(size)
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error for image {url}: {req_err}")
    except Exception as e:
        logging.error(f"Error processing image {url}: {e}")
    return None

def load_images(df):
    """
    Applies `preprocess_image` to every image URL in the DataFrame.
    Drops rows where image processing fails.
    """
    #df['image'] = df['image_url'].apply(preprocess_image)
    df.loc[:, 'image'] = df['image_url'].apply(preprocess_image)
    return df[df['image'].notnull()]  # Keep only rows with successfully loaded images  

#got to bring pixel_values here
#then run the test of it to verify, eventuakky re run other files .. and then evaluator againa nd continue.



def get_vilt_dataloader(data_path, batch_size, include_metadata=True):
    train_path = os.path.join(data_path, "train_data.tsv")
    val_path = os.path.join(data_path, "val_data.tsv")
    test_path = os.path.join(data_path, "test_data.tsv")

    train_df = pd.read_csv(train_path, sep="\t")
    val_df = pd.read_csv(val_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    # Preprocessing
    train_df = load_images(train_df)
    val_df = load_images(val_df)
    test_df = load_images(test_df)

    # Standardize metadata
    metadata_columns = ["num_comments", "score", "upvote_ratio"]
    scaler = StandardScaler()
    train_df[metadata_columns] = scaler.fit_transform(train_df[metadata_columns])
    val_df[metadata_columns] = scaler.transform(val_df[metadata_columns])
    test_df[metadata_columns] = scaler.transform(test_df[metadata_columns])

    # Load Dataloader
    train_loader = DataLoader(
        train_df, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_df, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_df, batch_size=batch_size, collate_fn=collate_fn)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }, None  # Replace None with class weights if applicable
