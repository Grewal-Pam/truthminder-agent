import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pickle
from utils.logger import setup_logger

logger = setup_logger("data_analysis.log")

class DataAnalysis:
    def __init__(self, output_dir="analysis_results", logger=None):
        self.output_dir = output_dir
        self.logger = logger or setup_logger("data_analysis.log")
        os.makedirs(self.output_dir, exist_ok=True)

    def check_missing_values(self, df, logger=None):
        """
        Check for missing values in the dataset and log the results.

        Args:
            df (pd.DataFrame): The dataframe to analyze.
            logger (logging.Logger): Logger to use for logging.

        Returns:
            pd.Series: Count of missing values per column.
        """
        logger = logger or self.logger
        if df.empty:
            logger.warning("Dataframe is empty. No missing values to check.")
            return pd.Series(dtype=int)

        missing_values = df.isnull().sum()
        logger.info("Missing Values Per Column:\n%s", missing_values.to_string())
        return missing_values

    def check_class_imbalance(self, df, label_column, logger=None):
        """
        Check for class imbalance in the label column.

        Args:
            df (pd.DataFrame): The dataframe to analyze.
            label_column (str): The column containing labels.
            logger (logging.Logger): Logger to use for logging.

        Returns:
            pd.Series: Distribution of classes.
            float: Imbalance ratio.
        """
        logger = logger or self.logger
        if label_column not in df.columns:
            logger.warning(f"Label column {label_column} not found in dataframe.")
            return None, None

        distribution = df[label_column].value_counts()
        imbalance_ratio = distribution.max() / distribution.min() if distribution.min() > 0 else np.inf
        logger.info(f"Class Distribution for {label_column}:\n{distribution.to_string()}")
        logger.info(f"Imbalance Ratio for {label_column}: {imbalance_ratio:.2f}")
        return distribution, imbalance_ratio

    def normalize_metadata(self, df, metadata_columns, scaler_path=None):
        """
        Normalize metadata columns and optionally save the scaler.

        Args:
            df (pd.DataFrame): The dataframe containing metadata.
            metadata_columns (list): List of metadata columns to normalize.
            scaler_path (str, optional): Path to save the scaler. Defaults to None.

        Returns:
            pd.DataFrame: The dataframe with normalized metadata.
        """
        if not all(col in df.columns for col in metadata_columns):
            missing_cols = [col for col in metadata_columns if col not in df.columns]
            logger.warning(f"Missing metadata columns: {missing_cols}")
            return df

        df = df.dropna(subset=metadata_columns)
        scaler = StandardScaler()
        df[metadata_columns] = scaler.fit_transform(df[metadata_columns])
        logger.info(f"Metadata columns normalized: {metadata_columns}")

        if scaler_path:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to: {scaler_path}")

        return df

    def plot_histogram(self, df, column):
        """
        Plot and save a histogram for a specified column.

        Args:
            df (pd.DataFrame): The dataframe containing the column.
            column (str): The column to plot.
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataframe.")
            return

        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        output_path = os.path.join(self.output_dir, f"histogram_{column}.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Histogram saved for column {column}: {output_path}")

    def plot_class_distribution(self, df, label_column):
        """
        Plot and save a bar chart for class distribution.

        Args:
            df (pd.DataFrame): The dataframe containing the label column.
            label_column (str): The label column to analyze.
        """
        if label_column not in df.columns:
            logger.warning(f"Label column {label_column} not found in dataframe.")
            return

        plt.figure(figsize=(8, 6))
        df[label_column].value_counts().plot(kind='bar')
        plt.title(f"Class Distribution for {label_column}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        output_path = os.path.join(self.output_dir, f"class_distribution_{label_column}.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Class distribution plot saved for column {label_column}: {output_path}")

    def check_outliers(self, df, columns):
        """
        Check for outliers in specified columns and log results.

        Args:
            df (pd.DataFrame): The dataframe to analyze.
            columns (list): List of columns to check for outliers.

        Returns:
            dict: A dictionary with outlier counts and indices for each column.
        """
        outlier_info = {}
        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in dataframe.")
                continue

            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_info[column] = {
                "count": outliers.shape[0],
                "indices": outliers.index.tolist()
            }
            logger.info(f"Outliers in column {column}: {outlier_info[column]['count']}")
        return outlier_info

    def generate_summary_report(self, df):
        """
        Generate a summary report for the dataset and save it as a CSV file.

        Args:
            df (pd.DataFrame): The dataframe to summarize.

        Returns:
            pd.DataFrame: The summary report.
        """
        if df.empty:
            logger.warning("Cannot generate summary report for an empty dataframe.")
            return pd.DataFrame()

        summary = df.describe(include='all').transpose()
        output_path = os.path.join(self.output_dir, "dataset_summary.csv")
        summary.to_csv(output_path)
        logger.info(f"Dataset summary saved to: {output_path}")
        return summary
