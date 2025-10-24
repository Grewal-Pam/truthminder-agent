import asyncio
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import config
from utils.logger import setup_logger
from transformers import CLIPProcessor
from PIL import Image
import requests
from io import BytesIO
import logging
import os
from multiprocessing import Pool, cpu_count
import psutil
from aiohttp import ClientSession
import os
import pickle
import pytesseract


logger = setup_logger("pre_processing_log") #, log_dir=config.log_dir, sampled=False


# Initialize the CLIPProcessor globally
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

async def validate_url(url, session, semaphore):
    """
    Validate a single URL asynchronously.

    Args:
        url (str): The URL to validate.
        session (ClientSession): The aiohttp session.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.

    Returns:
        tuple: URL and validation status (True/False).
    """
    async with semaphore:
        try:
            async with session.head(url, timeout=10, allow_redirects=True) as response:
                if response.status == 200 and "image" in response.headers.get("Content-Type", ""):
                    return url, True
                else:
                    logger.warning(f"URL does not point to an image: {url}")
                    return url, False
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return url, False

async def check_urls(urls, session, semaphore):
    """
    Validate a batch of URLs asynchronously.

    Args:
        urls (list): List of URLs to validate.
        session (ClientSession): The aiohttp session.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.

    Returns:
        list: List of tuples (URL, validation status).
    """
    tasks = [validate_url(url, session, semaphore) for url in urls]
    return await asyncio.gather(*tasks)

async def run_batches(urls, batch_size=50):
    """
    Process URLs in batches asynchronously.

    Args:
        urls (list): List of URLs to validate.
        batch_size (int): Number of URLs to process in each batch.

    Returns:
        list: List of valid URLs.
    """
    valid_urls = []
    semaphore = asyncio.Semaphore(100)  # Limit concurrency to 100 requests
    async with ClientSession() as session:
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            results = await check_urls(batch, session, semaphore)
            valid_urls.extend([url for url, is_valid in results if is_valid])
            logger.info(f"Processed {i + len(batch)} URLs. Valid URLs so far: {len(valid_urls)}")
    return valid_urls

def validate_urls_in_dataframe(df, image_column="image_url", batch_size=50):
    """
    Validate all URLs in a DataFrame asynchronously.

    Args:
        df (pd.DataFrame): DataFrame containing URLs to validate.
        image_column (str): Column name containing image URLs.
        batch_size (int): Number of URLs to process in each batch.

    Returns:
        pd.DataFrame: DataFrame with only valid URLs.
    """
    logger.info(f"Validating {len(df)} URLs in column '{image_column}'...")
    urls = df[image_column].tolist()

    # Run async validation
    valid_urls = asyncio.run(run_batches(urls, batch_size=batch_size))

    # Filter DataFrame
    valid_df = df[df[image_column].isin(valid_urls)].reset_index(drop=True)
    logger.info(f"Validation complete. Remaining rows: {len(valid_df)}")
    return valid_df

def preprocess_alt(df, dataset_name, metadata_columns, image_column="image_url", chunk_size=1000, batch_size=50):
    log_dataset_stats(df, f"Before preprocessing - {dataset_name}")

    # Validate URLs
    validated_file_path = f"data/{dataset_name}_validated.tsv"
    if os.path.exists(validated_file_path):
        logger.info(f"Loading pre-validated dataset: {validated_file_path}")
        df = pd.read_csv(validated_file_path, sep="\t")
    else:
        df = validate_urls_in_dataframe(df, image_column=image_column, batch_size=batch_size)
        df.to_csv(validated_file_path, sep="\t", index=False)
        logger.info(f"Validated dataset saved to: {validated_file_path}")

    # # Continue preprocessing
    # df = normalize_metadata(df, metadata_columns)
    # save_intermediate_data(df, f"{dataset_name}_normalized.tsv")
    # df = handle_missing_values(df, ["clean_title", image_column, "2_way_label", "3_way_label"])
    # save_intermediate_data(df, f"{dataset_name}_cleaned.tsv")
    # df = compute_pixel_values_in_chunks(df, chunk_size=chunk_size, batch_size=batch_size, image_column=image_column)
    # save_intermediate_data(df, f"{dataset_name}_processed.tsv")
    # log_dataset_stats(df, f"After preprocessing - {dataset_name}")
    # return df
     # Normalize metadata
    normalized_file_path = f"data/{dataset_name}_normalized.tsv"
    if os.path.exists(normalized_file_path):
        logger.info(f"Loading normalized dataset: {normalized_file_path}")
        df = pd.read_csv(normalized_file_path, sep="\t")
    else:
        df = normalize_metadata(df, metadata_columns)
        df.to_csv(normalized_file_path, sep="\t", index=False)
        logger.info(f"Normalized dataset saved to: {normalized_file_path}")

    # Handle missing values
    cleaned_file_path = f"data/{dataset_name}_cleaned.tsv"
    if os.path.exists(cleaned_file_path):
        logger.info(f"Loading cleaned dataset: {cleaned_file_path}")
        df = pd.read_csv(cleaned_file_path, sep="\t")
    else:
        df = handle_missing_values(df, ["clean_title", image_column, "2_way_label", "3_way_label"])
        df.to_csv(cleaned_file_path, sep="\t", index=False)
        logger.info(f"Cleaned dataset saved to: {cleaned_file_path}")

    # Compute pixel values
    # processed_file_path = f"data/{dataset_name}_processed.tsv"
    # if os.path.exists(processed_file_path):
    #     logger.info(f"Loading processed dataset: {processed_file_path}")
    #     df = pd.read_csv(processed_file_path, sep="\t")
    # else:
    #     df = compute_pixel_values_in_chunks(df, chunk_size=chunk_size, batch_size=batch_size, image_column=image_column, debug_rows=1)
    #     df.to_csv(processed_file_path, sep="\t", index=False)
    #     logger.info(f"Processed dataset saved to: {processed_file_path}")

    log_dataset_stats(df, f"After preprocessing - {dataset_name}")
    return df

###execute this####

# Preprocess image function
def preprocess_image(url, size=(224, 224)):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': url,  # Simulate browser behavior
    })
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raise for bad status codes
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Check for empty, black, white, or placeholder images
        if img.size == (0, 0):
            logger.error(f"Empty image at URL: {url}")
            return None
        if is_placeholder_image(img):
            logger.warning(f"Placeholder image at URL: {url}")
            return None

        return img.resize(size)
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

# Placeholder detection function
def is_placeholder_image(img):
    """
    Detect if an image is a placeholder or invalid based on visual and OCR analysis.
    """
    try:
        # Convert to grayscale for easier analysis
        grayscale = img.convert("L")

        # Check for all black or all white images
        extrema = grayscale.getextrema()
        if extrema == (0, 0):  # All black
            logger.warning("Image is completely black.")
            return True
        if extrema == (255, 255):  # All white
            logger.warning("Image is completely white.")
            return True

        # Check for low variance (mostly uniform color)
        histogram = grayscale.histogram()
        total_pixels = sum(histogram)
        max_pixel_count = max(histogram)
        if max_pixel_count / total_pixels > 0.98:  # Mostly uniform color
            logger.warning("Image has low variance and is likely a placeholder.")
            return True

        # OCR-based placeholder detection
        try:
            text = pytesseract.image_to_string(img)
            #logger.debug(f"OCR extracted text: {text}")
            placeholder_keywords = [
                "The image you are requesting does not exist",
                "no longer available",
                "image unavailable",
                "broken link",
                "imgur",
                "tinypic",
                "not found",
                "error",
                "HuffPost",
            ]
            for keyword in placeholder_keywords:
                if keyword.lower() in text.lower():
                    #logger.warning(f"Detected placeholder keyword in image text: {keyword}")
                    return True
        except Exception as e:
            logger.warning(f"OCR error while processing image: {e}")

        # Color/Pattern-based checks for logos or other characteristics
        # Example: Check for specific dominant colors (e.g., typical placeholder gray)
        dominant_color = img.resize((1, 1)).getpixel((0, 0))
        if dominant_color in [(128, 128, 128), (192, 192, 192), (240, 240, 240)]:
            logger.warning(f"Detected placeholder image based on dominant color: {dominant_color}")
            return True

        # Optional: Detect specific patterns/logos using more advanced methods (requires OpenCV)
        # For example, detecting "imgur" or "tinypic" logos
        # Implement this only if needed and feasible for your setup

    except Exception as e:
        logger.error(f"Error during placeholder detection: {e}")
        return True  # Treat as placeholder if detection fails

    return False



# Handle missing values
def handle_missing_values(df, columns, method="drop", fill_value=0.0):
    initial_len = len(df)
    if method == "drop":
        df = df.dropna(subset=columns)
        logger.info(f"Dropped {initial_len - len(df)} rows due to missing values.")
    elif method == "fill":
        for col in columns:
            df[col] = df[col].fillna(fill_value)
        logger.info(f"Filled missing values in columns: {columns}.")
    return df

# Normalize metadata
def normalize_metadata(df, metadata_columns, scaler_path="scaler.pkl"):
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded existing scaler for metadata normalization.")
    else:
        scaler = StandardScaler()
        df[metadata_columns] = scaler.fit_transform(df[metadata_columns])
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler for future use at {scaler_path}.")
    return df

# Log dataset statistics
def log_dataset_stats(df, name):
    logger.info(f"Dataset: {name}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Statistical Summary:\n{df.describe(include='all')}")

# Preprocess function
def preprocess(df, dataset_name, metadata_columns, image_column="image_url", scaler_path="scaler.pkl", batch_size=1000):
    log_dataset_stats(df, f"Before preprocessing - {dataset_name}")
    processed_batches = []
    df = df[df[image_column].notnull()]  # Drop rows with missing image URLs
    # Load the existing scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Ensure it is computed before preprocessing.")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(df) - 1) // batch_size + 1}")
        
        try:
            # Validate URLs
            valid_urls = [url for url in batch[image_column] if preprocess_image(url)]
            batch = batch[batch[image_column].isin(valid_urls)]

            # Normalize Metadata
            batch[metadata_columns] = scaler.transform(batch[metadata_columns])

            # Handle Missing Values
            batch = handle_missing_values(batch, columns=["clean_title", image_column, "2_way_label", "3_way_label", *metadata_columns])

            processed_batches.append(batch)

            # Save each batch
            batch_path = f"data/{dataset_name}_batch_{i // batch_size + 1}.tsv"
            batch.to_csv(batch_path, sep="\t", index=False)
            logger.info(f"Batch {i // batch_size + 1} saved to {batch_path}.")
        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1}: {e}")

    # Combine processed batches and save final dataset
    processed_df = pd.concat(processed_batches, ignore_index=True)
    final_path = f"data/{dataset_name}_preprocessed.tsv"
    processed_df.to_csv(final_path, sep="\t", index=False)
    logger.info(f"Final preprocessed dataset saved to {final_path}. Total rows: {len(processed_df)}")
    return processed_df

def load_and_preprocess_datasets(train_path, validate_path, test_path, metadata_columns, image_column="image_url"):
    train_df = pd.read_csv(train_path, sep="\t")#.sample(n=10000, random_state=42)  # Sample 10,000 rows
    val_df = pd.read_csv(validate_path, sep="\t")#.sample(n=1000, random_state=42)  # Sample 1,000 rows
    test_df = pd.read_csv(test_path, sep="\t")#.sample(n=1000, random_state=42)  # Sample 1,000 rows

    # Step 1: Fit Scaler on Full Training Data
    scaler_path = "scaler.pkl"
    if not os.path.exists(scaler_path):
        logger.info("Fitting scaler on the full training dataset...")
        scaler = StandardScaler()
        scaler.fit(train_df[metadata_columns])  # Fit on training metadata columns
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
    else:
        logger.warning(f"Scaler file {scaler_path} already exists. It will be reused for normalization.")

    # Preprocess datasets
    train_df = preprocess(train_df, "train", metadata_columns, image_column, scaler_path, batch_size=1000)
    val_df = preprocess(val_df, "validate", metadata_columns, image_column, scaler_path, batch_size=1000)
    test_df = preprocess(test_df, "test", metadata_columns, image_column, scaler_path, batch_size=1000)

    # Save preprocessed datasets
    train_df.to_csv("data/TRAIN DATA/train_preprocessed.tsv", sep="\t", index=False)
    val_df.to_csv("data/VALIDATE DATA/validate_preprocessed.tsv", sep="\t", index=False)
    test_df.to_csv("data/TEST DATA/test_preprocessed.tsv", sep="\t", index=False)

    logger.info("Sample datasets preprocessed and saved.")
    return train_df, val_df, test_df

###execute until this####


# def preprocess(df, dataset_name, metadata_columns, image_column="image_url", chunk_size=1000):
#     """
#     Preprocess a dataset: normalize metadata, handle missing values, and compute pixel values.
    
#     Args:
#         df (pd.DataFrame): The input dataset.
#         dataset_name (str): Name of the dataset (e.g., "train", "validate", "test").
#         metadata_columns (list): List of metadata column names.
#         image_column (str): Column containing image URLs or paths.
#         chunk_size (int): Size of chunks to process images in.
        
#     Returns:
#         pd.DataFrame: Preprocessed dataset.
#     """
#     log_dataset_stats(df, f"Before preprocessing - {dataset_name}")
#     # Validate URLs
#     df["valid_url"] = df[image_column].apply(validate_image_url)
#     df = df[df["valid_url"]].drop(columns=["valid_url"])
#     logger.info(f"Filtered invalid URLs. Remaining rows: {len(df)}")

#     df = normalize_metadata(df, metadata_columns)
#     save_intermediate_data(df, f"{dataset_name}_normalized.parquet")
#     df = handle_missing_values(df, ["clean_title", image_column, "2_way_label", "3_way_label"])
#     save_intermediate_data(df, f"{dataset_name}_cleaned.parquet")
#     df = compute_pixel_values_in_chunks(df, chunk_size=chunk_size, image_column=image_column)
#     save_intermediate_data(df, f"{dataset_name}_processed.parquet")
#     log_dataset_stats(df, f"After preprocessing - {dataset_name}")
#     return df

# Log dataset stats
# def log_dataset_stats_alt(df, stage):
#     logger.info(f"{stage} dataset stats:")
#     logger.info(f"Total rows: {len(df)}")
#     logger.info(f"Columns: {list(df.columns)}")
#     logger.info(f"Missing values: {df.isnull().sum().to_dict()}")
#     logger.info(f"First 3 rows:\n{df.head(3)}")

# Save and load intermediate data
def save_intermediate_data(df, path):
    df.to_parquet(path, index=False)
    logger.info(f"Intermediate data saved at: {path}")


def load_intermediate_data(path):
    logger.info(f"Loading intermediate data from: {path}")
    return pd.read_parquet(path)


# Log resource usage
def log_resource_usage():
    logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
    logger.info(f"Memory Usage: {psutil.virtual_memory().percent}% of {psutil.virtual_memory().total // (1024 ** 2)} MB")


def compute_pixel_values_in_chunks(df, chunk_size=500, batch_size=50, image_column="image_url", debug_rows=None):
    """
    Process pixel values in smaller batches within chunks to avoid memory overload.
    Includes debugging option to process only the first N rows.
    """
    # If debugging, limit to the first debug_rows rows
    if debug_rows and len(df) > debug_rows:
        logger.info(f"Debugging enabled: Processing only the first {debug_rows} rows.")
        df = df.head(debug_rows)

    logger.info(f"Processing pixel values in chunks of {chunk_size}, batch size: {batch_size}...")
    processed_chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]

        try:
            # Process in smaller batches
            processed_chunk = process_batches(chunk, batch_size, image_column)
            processed_chunks.append(processed_chunk)

            # Save intermediate checkpoints
            checkpoint_path = f"processed_chunk_{i}.parquet"
            processed_chunk.to_parquet(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error processing chunk {i // chunk_size}: {e}")
            break

    return pd.concat(processed_chunks, ignore_index=True) if processed_chunks else df



def process_batches(df, batch_size, image_column):
    """
    Process images in smaller batches to reduce memory load.
    """
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch_results = process_batch(batch[image_column].tolist())
        
        # Debugging the results structure
        logger.info(f"Batch {i // batch_size + 1}: Type of batch_results: {type(batch_results)}")
        logger.info(f"Batch {i // batch_size + 1}: Length of batch_results: {len(batch_results)}")
        logger.info(f"Batch {i // batch_size + 1}: Sample of batch_results: {batch_results[:3]}")

        # Extract pixel_values from results
        pixel_values = [r["pixel_values"] for r in batch_results if r is not None]

        # Validate batch size consistency
        if len(pixel_values) != len(batch):
            logger.warning(f"Batch size mismatch: Expected {len(batch)}, got {len(pixel_values)}. Skipping this batch.")
            continue

        results.extend(pixel_values)
        logger.info(f"Processed batch {i // batch_size + 1} of {len(df) // batch_size + 1}")
        logger.debug(f"First few results from process_batch: {results[:3]}")

    # Convert results to a DataFrame
    batch_results_df = pd.DataFrame(results, columns=["pixel_values"])
    df = df.reset_index(drop=True)

    # Validate final DataFrame size
    if len(batch_results_df) != len(df):
        logger.error("Mismatch between results and DataFrame size after processing batches.")
        raise ValueError("Pixel values do not match the DataFrame size.")

    df["pixel_values"] = batch_results_df["pixel_values"]
    return df.dropna(subset=["pixel_values"]).reset_index(drop=True)



def process_batch(urls):
    """
    Process a single batch of URLs to compute pixel values.
    """
    with Pool(cpu_count() // 2) as pool:  # Limit the number of workers
        results = pool.map(process_image_parallel, urls)

    # Debugging results structure
    logger.info(f"Type of results: {type(results)}")
    logger.info(f"Length of results: {len(results)}")
    logger.info(f"Sample results: {results[:3]}")

    # Filter out invalid results
    valid_results = [{"pixel_values": r["pixel_values"]} for r in results if r and "pixel_values" in r]

    # Debugging: Log results for inspection
    logger.debug(f"Batch processing complete. Total valid results: {len(valid_results)}")
    return valid_results


# Image preprocessing
def preprocess_image___(url, size=(224, 224)):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img.resize(size)
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error for image {url}: {req_err}")
    except Exception as e:
        logger.error(f"Error processing image {url}: {e}")
    return None

# Compute pixel values with multiprocessing
def compute_pixel_values(df, image_column="image_url"):
    logger.info("Starting pixel value computation using multiprocessing...")
    log_resource_usage()

    results = []
    with Pool(max(1, cpu_count() // 2)) as pool:
        results = pool.map(process_image_parallel, df[image_column].tolist())

    # Extract results and update DataFrame
    #df["image"] = [result["image"] for result in results]
    df["pixel_values"] = [result["pixel_values"] for result in results]

    # Drop rows with invalid pixel values
    df = df[df["pixel_values"].notnull()].reset_index(drop=True)

    logger.info(f"Computed pixel values for {len(df)} valid rows.")
    log_resource_usage()
    return df


def process_image_parallel(url):
    try:
        img = preprocess_image(url)
        if img is not None:
            pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0].tolist()
            logger.info(f"Processed image from {url}, sample pixel_values: {pixel_values[:5]}")
            return {"pixel_values": pixel_values}
    except Exception as e:
        logger.warning(f"Failed to process image {url}: {e}")
    return None




# def validate_image_url(url):
#     """
#     Check if an image URL is accessible and valid.
#     """
#     session = requests.Session()
#     session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

#     try:
#         response = session.head(url, timeout=10, allow_redirects=True)  # Allow redirects
#         if response.status_code == 200:
#             content_type = response.headers.get("Content-Type", "")
#             if "image" in content_type:  # Validate that it's an image
#                 return True
#             else:
#                 logger.warning(f"URL does not point to an image: {url}")
#                 return False
#         else:
#             logger.warning(f"Invalid status code {response.status_code} for URL: {url}")
#             return False
#     except requests.RequestException as e:
#         logger.error(f"Error validating URL {url}: {e}")
#         return False


def compute_pixel_values_alt(df, image_column="image_url", size=(224, 224)):
    """
    Processes the dataset to compute pixel values for images and drops invalid rows.

    Args:
        df (pd.DataFrame): Input DataFrame with an image URL column.
        image_column (str): The column containing image URLs.
        size (tuple): Size to which images are resized.

    Returns:
        pd.DataFrame: DataFrame with a new `pixel_values` column for valid images.
    """
    def process_image(url):
        """
        Preprocess and compute pixel values for a single image URL.
        """
        img = preprocess_image(url, size=size)
        if img is not None:
            try:
                pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0].tolist()
                return img, pixel_values
            except Exception as e:
                logger.error(f"Error computing pixel values for image {url}: {e}")
        return None

    logger.info("Starting pixel value computation...")
    #df['pixel_values'] = df[image_column].apply(process_image)
    # Process each image and retain both the original image and pixel values
    df[['image', 'pixel_values']] = df[image_column].apply(
        lambda url: pd.Series(process_image(url))
    )
    df = df[df['pixel_values'].notnull()].reset_index(drop=True)  # Drop rows with invalid images
    logger.debug(f"Example pixel values: {df['pixel_values'].iloc[0]}")
    logger.info(f"Computed pixel values for {len(df)} valid rows.")
    return df


# def handle_missing_values(df, columns, method="drop", fill_value=0.0):
#     initial_len = len(df)
#     if method == "drop":
#         df = df.dropna(subset=columns)
#         logger.info(f"Dropped {initial_len - len(df)} rows due to missing values.")
#     elif method == "fill":
#         for col in columns:
#             df[col] = df[col].fillna(fill_value)
#         logger.info(f"Filled missing values in columns: {columns}.")
#     return df


# Normalize metadata with optional scaler saving/loading
# def normalize_metadata(df, metadata_columns, scaler_path="scaler.pkl"):
#     if os.path.exists(scaler_path):
#         with open(scaler_path, "rb") as f:
#             scaler = pickle.load(f)
#         logger.info("Loaded existing scaler for metadata normalization.")
#     else:
#         scaler = StandardScaler()
#         df[metadata_columns] = scaler.fit_transform(df[metadata_columns])
#         with open(scaler_path, "wb") as f:
#             pickle.dump(scaler, f)
#         logger.info(f"Saved scaler for future use at {scaler_path}.")
#     return df


def split_dataset(df, test_size=0.2, val_size=0.25, seed=42):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed)
    logger.info(f"Split dataset into train ({len(train_df)}), val ({len(val_df)}), and test ({len(test_df)}) sets.")
    return train_df, val_df, test_df


# Usage in `tests/test_trainer.py`
# Ensure the following:
# - Import `compute_pixel_values` in your test file.
# - Call `compute_pixel_values(df)` before creating datasets for training.

# Example:
# df = compute_pixel_values(df, image_column="image_url")
