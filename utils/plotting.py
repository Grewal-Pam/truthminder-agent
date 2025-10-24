import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score


class Plotting:
    def __init__(self, output_dir="results/images"):
        """
        Initialize the Plotting class and create the images directory if it doesn't exist.

        Args:
            output_dir (str): Directory where plots will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_plot(self, fig, filename):
        """
        Save a plot to the specified output directory.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): The name of the file to save the plot.
        """
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Plot saved: {filepath}")

    def plot_histogram(self, data, column, title=None):
        """
        Plot a histogram for the specified column.

        Args:
            data (pd.DataFrame): The dataframe containing the data.
            column (str): The column to plot.
            title (str, optional): Title for the plot. Defaults to None.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(title if title else f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        self.save_plot(fig, f"histogram_{column}.png")

    def plot_class_distribution(self, data, label_column, title=None):
        """
        Plot a bar chart for class distribution.

        Args:
            data (pd.DataFrame): The dataframe containing the label column.
            label_column (str): The label column to analyze.
            title (str, optional): Title for the plot. Defaults to None.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        data[label_column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(title if title else f"Class Distribution for {label_column}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        self.save_plot(fig, f"class_distribution_{label_column}.png")

    def plot_confusion_matrix(self, y_true, y_pred, labels, save_as, title=None):
        """
        Plot and save the confusion matrix.

        Args:
            y_true (list or np.ndarray): True labels.
            y_pred (list or np.ndarray): Predicted labels.
            labels (list): List of class labels.
            save_as (str): Filename to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        self.save_plot(fig, save_as)

    def plot_classification_report(self, y_true, y_pred, labels, save_as, title=None):
        """
        Generate and save a plot for the classification report.

        Args:
            y_true (list or np.ndarray): True labels.
            y_pred (list or np.ndarray): Predicted labels.
            labels (list): List of class labels.
            save_as (str): Filename to save the plot.
        """
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        metrics = ["precision", "recall", "f1-score"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, metric in enumerate(metrics):
            values = [report[str(label)][metric] for label in labels]
            ax.bar(np.arange(len(labels)) + idx * 0.2, values, width=0.2, label=metric)

        ax.set_xticks(np.arange(len(labels)) + 0.3)
        ax.set_xticklabels(labels)
        ax.set_title("Classification Report")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Scores")
        ax.legend()
        self.save_plot(fig, save_as)

    def plot_precision_recall_curve(
        self, y_true, y_scores, save_as, title=None, labels=None
    ):
        """
        Plot and save the precision-recall curve for multiclass or binary classification.

        Args:
        y_true (list or np.ndarray): True labels.
        y_scores (np.ndarray): Predicted scores or probabilities.
        save_as (str): Filename to save the plot.
        title (str, optional): Title for the plot.
        labels (list, optional): List of class labels for multiclass problems.
        """
        # Check if labels are provided for multiclass
        if labels is not None and len(labels) > 2:  # Multiclass
            # Binarize labels for one-vs-rest (OvR) approach
            y_true_bin = label_binarize(y_true, classes=range(len(labels)))
            y_scores = np.array(y_scores)

            fig, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_scores[:, i]
                )
                ap_score = average_precision_score(y_true_bin[:, i], y_scores[:, i])
                ax.plot(recall, precision, label=f"Class {label} (AP = {ap_score:.2f})")

            ax.set_title(title if title else "Multiclass Precision-Recall Curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
        else:  # Binary classification
            if labels is None:  # Default binary labels
                labels = [0, 1]
            y_scores = np.array(y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, label="Precision-Recall Curve")
            ax.set_title(title if title else "Binary Precision-Recall Curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
        self.save_plot(fig, save_as)

    def plot_roc_curve(self, y_true, y_scores, save_as, title=None, labels=None):
        """
        Plot and save the ROC curve for multiclass or binary classification.

        Args:
            y_true (list or np.ndarray): True labels.
            y_scores (np.ndarray): Predicted scores or probabilities.
            save_as (str): Filename to save the plot.
            title (str, optional): Title for the plot.
            labels (list, optional): List of class labels for multiclass problems.
        """
        y_scores = np.array(y_scores)  # Convert to NumPy array if not already

        # Binary classification case
        if labels is None or len(labels) <= 2:
            if y_scores.ndim != 1:
                raise ValueError(
                    f"Expected y_scores to be 1D for binary classification, but got shape {y_scores.shape}."
                )
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, marker=".", label=f"ROC Curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title(title if title else "ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
        else:  # Multiclass case
            if y_scores.ndim != 2 or y_scores.shape[1] != len(labels):
                raise ValueError(
                    f"Expected y_scores to be 2D with shape (n_samples, n_classes) for multiclass. "
                    f"Got shape {y_scores.shape}."
                )
            y_true_bin = label_binarize(y_true, classes=range(len(labels)))
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {label} (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title(title if title else "Multiclass ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()

        self.save_plot(fig, save_as)
