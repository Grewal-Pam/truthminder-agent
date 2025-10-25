import asyncio
from aiohttp import ClientSession
import pandas as pd
import nest_asyncio
import os
import logging

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    filename="dataset_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Define file paths dynamically
BASE_DIR = "data"
TRAIN_INPUT_FILE = os.path.join(
    BASE_DIR, "multimodal_only_samples", "multimodal_train.tsv"
)
TEST_INPUT_FILE = os.path.join(
    BASE_DIR, "multimodal_only_samples", "multimodal_test_public.tsv"
)
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "TRAIN DATA")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "TEST DATA")
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)


# Async function to fetch URL status
async def fetch(url, session, semaphore):
    async with semaphore:  # Control concurrency
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with session.head(
                url, allow_redirects=True, timeout=10, headers=headers
            ) as response:
                return url, response.status == 200
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return url, False


# Process batches of URLs
async def check_urls(batch, session, semaphore):
    tasks = [
        asyncio.create_task(fetch(url, session, semaphore))
        for url in batch
        if pd.notna(url)
    ]
    return await asyncio.gather(*tasks)


# Run batches asynchronously
async def run_batches(urls, batch_size=50):
    results = []
    semaphore = asyncio.Semaphore(100)  # Adjust based on system capabilities
    async with ClientSession() as session:
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            results.extend(await check_urls(batch, session, semaphore))
            logger.info(f"Processed {i + len(batch)} URLs")
    return results


# Main processing function
async def process_dataset(input_file, output_dir):
    # Load dataset
    try:
        data = pd.read_csv(input_file, sep="\t")
        logger.info(f"Loaded dataset: {input_file}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Clean dataset
    data.dropna(subset=["image_url", "clean_title"], inplace=True)
    data["author"].fillna("Unknown", inplace=True)
    data["domain"].fillna("No domain", inplace=True)
    data["created_utc"] = pd.to_datetime(data["created_utc"], unit="s")
    data.drop_duplicates(inplace=True)

    logger.info(f"Cleaned dataset: {data.shape}")

    # Check URL accessibility
    urls = data["image_url"].tolist()
    accessible_results = await run_batches(urls)

    # Update DataFrame with accessibility info
    data["url_accessible"] = False
    for url, accessible in accessible_results:
        data.loc[data["image_url"] == url, "url_accessible"] = accessible

    logger.info(f"Accessible URLs: {data['url_accessible'].sum()} / {len(data)}")

    # Save filtered datasets
    accessible_data = data[data["url_accessible"]]
    inaccessible_data = data[~data["url_accessible"]]

    accessible_data.to_csv(
        os.path.join(output_dir, "train_data.tsv"), sep="\t", index=False
    )
    # accessible_data.to_csv(os.path.join(output_dir, "test_data.tsv"), sep="\t", index=False)
    inaccessible_data.to_csv(
        os.path.join(output_dir, "train_data_inaccessible.tsv"), sep="\t", index=False
    )
    # inaccessible_data.to_csv(os.path.join(output_dir, "test_data_inaccessible.tsv"), sep="\t", index=False)

    logger.info("Saved accessible and inaccessible datasets.")


# Run the script
if __name__ == "__main__":
    asyncio.run(process_dataset(TRAIN_INPUT_FILE, TRAIN_OUTPUT_DIR))
    # asyncio.run(process_dataset(TEST_INPUT_FILE, TEST_OUTPUT_DIR))
