import datetime
import json
import logging
import os


class ExperimentFolderManager:
    def __init__(self, base_dir, model_name, timestamp=None):
        """
        Initialize the folder manager for a specific experiment.

        Args:
            base_dir (str): Base directory for experiments.
            model_name (str): Name of the model (e.g., 'flava', 'vilt', 'clip').
            timestamp (str): Timestamp for the experiment folder (default is current time).
        """
        self.timestamp = timestamp or datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self.base_dir = os.path.join(base_dir, model_name.upper(), self.timestamp)
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.metrics_dir = os.path.join(self.base_dir, "metrics")
        self.images_dir = os.path.join(self.base_dir, "images")
        self.models_dir = os.path.join(self.base_dir, "models")

        # Create all necessary directories
        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def get_log_file(self, model_name):
        """
        Generate the log file path.

        Args:
            model_name (str): Name of the model.

        Returns:
            str: Full path to the log file.
        """
        return os.path.join(self.logs_dir, f"{model_name}_experiment.log")

    def save_metadata(self, metadata):
        """
        Save run metadata to the output directory.

        Args:
            metadata (dict): Metadata dictionary to save.
        """
        metadata_path = os.path.join(self.base_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata saved at: {metadata_path}")
