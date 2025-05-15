import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from utils.folder_manager import ExperimentFolderManager

class PlottingManager:
    def __init__(self, model_name, folder_manager):
        """
        Initialize the PlottingManager with a model-specific folder structure.
        
        Args:
            model_name (str): Name of the model (e.g., 'flava', 'vilt', 'clip').
            output_dir (str): Base directory for saving results.
        """
        #self.base_dir = os.path.join(output_dir, model_name)
        self.image_dir = folder_manager.images_dir
        self.metric_dir = folder_manager.metrics_dir

        # Create directories if they don't exist
        # os.makedirs(self.image_dir, exist_ok=True)
        # os.makedirs(self.metric_dir, exist_ok=True)
        # os.makedirs(self.log_dir, exist_ok=True)

    def save_plot(self, fig, filename):
        filepath = os.path.join(self.image_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Plot saved: {filepath}")

    def save_metrics(self, metrics, filename):
        """
        Save metrics to a JSON file in the metric directory.

        Args:
            metrics (dict): Dictionary containing metrics.
            filename (str): The name of the file to save the metrics.
        """
        filepath = os.path.join(self.metric_dir, filename)
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved: {filepath}")
        
    # 1. Confusion Matrix
    def plot_confusion_matrix(self, y_true, y_pred, labels, save_as):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        self.save_plot(fig, save_as)

    # 2. Precision-Recall Curve
    def plot_precision_recall_curve(self, y_true, y_scores, save_as, labels=None):
        y_scores = np.array(y_scores)
        if labels and len(labels) > 2:  # Multiclass
            y_true_bin = label_binarize(y_true, classes=range(len(labels)))
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
                ap_score = auc(recall, precision)
                ax.plot(recall, precision, label=f"Class {label} (AP = {ap_score:.2f})")
            ax.set_title("Multiclass Precision-Recall Curve")
        else:  # Binary
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, label="Precision-Recall Curve")
            ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        self.save_plot(fig, save_as)

    # 3. ROC Curve
    def plot_roc_curve(self, y_true, y_scores, save_as, labels=None):
        """
        Plot and save the ROC curve for multiclass or binary classification.
        """
        y_scores = np.array(y_scores)

        # Ensure consistent lengths
        if len(y_true) != len(y_scores):
            raise ValueError(f"Inconsistent lengths: y_true={len(y_true)}, y_scores={y_scores.shape}")

        if labels and len(labels) > 2:  # Multiclass
            y_true_bin = label_binarize(y_true, classes=range(len(labels)))
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {label} (AUC = {roc_auc:.2f})")
            ax.set_title("Multiclass ROC Curve")
        else:  # Binary
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        self.save_plot(fig, save_as)


    # 4. Class Distribution
    def plot_class_distribution(self, data, label_column, save_as):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=data, x=label_column, ax=ax)
        ax.set_title(f"Class Distribution for {label_column}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        self.save_plot(fig, save_as)

    # 5. Loss and Accuracy Curves
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies, save_as):
        epochs = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, label="Training Loss")
        ax.plot(epochs, val_losses, label="Validation Loss")
        ax.plot(epochs, train_accuracies, label="Training Accuracy")
        ax.plot(epochs, val_accuracies, label="Validation Accuracy")
        ax.set_title("Training and Validation Curves")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metric Value")
        ax.legend()
        self.save_plot(fig, save_as)

    # 6. Classification Report
    def plot_classification_report(self, y_true, y_pred, labels, save_as):
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        metrics = ["precision", "recall", "f1-score"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, metric in enumerate(metrics):
            values = [report[label][metric] for label in labels]
            ax.bar(np.arange(len(labels)) + idx * 0.2, values, width=0.2, label=metric)
        ax.set_xticks(np.arange(len(labels)) + 0.3)
        ax.set_xticklabels(labels)
        ax.set_title("Classification Report")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Scores")
        ax.legend()
        self.save_plot(fig, save_as)

    def plot_lr_finder(self, lrs, losses, save_as):
        """
        Plots the LR Finder curve: Learning Rate vs Smoothed Loss.
        Uses same save logic as other plots via self.save_plot.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Smoothed Loss")
        ax.set_title("LR Finder: Loss vs Learning Rate")
        ax.grid(True)
        fig.tight_layout()

        self.save_plot(fig, save_as)



    # def plot_loss_curve():
    # # Plot the loss curve after training ends
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(training_history["epoch"], training_history["train_loss"], label="Train Loss", color="blue")
    #     plt.plot(training_history["epoch"], training_history["val_loss"], label="Validation Loss", color="red")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.title(f"Training and Validation Loss for {task_name}")
    #     plt.legend(loc="upper right")
    #     plt.grid(True)
    #     loss_curve_path = f"{folder_manager.models_dir}/{task_name.replace(' ', '_')}_loss_curve.png"
    #     plt.savefig(loss_curve_path)
    #     experiment_logger.info(f"Loss curve saved to {loss_curve_path}")
    #     plt.close()

    #     # Plot the metrics curve (accuracy or F1 score) after training ends
    #     val_accuracies = [metric["accuracy"] for metric in training_history["val_metrics"]]
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(training_history["epoch"], val_accuracies, label="Validation Accuracy", color="green")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.title(f"Validation Accuracy for {task_name}")
    #     plt.legend(loc="upper right")
    #     plt.grid(True)
    #     accuracy_curve_path = f"{folder_manager.models_dir}/{task_name.replace(' ', '_')}_accuracy_curve.png"
    #     plt.savefig(accuracy_curve_path)
    #     experiment_logger.info(f"Accuracy curve saved to {accuracy_curve_path}")
    #     plt.close()