from ast import arg
import json
import os
import sys
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import config
import torch
import pandas as pd
import numpy as np
import pickle
from local_datasets.pre_processing import compute_pixel_values, handle_missing_values, normalize_metadata, split_dataset, load_and_preprocess_datasets
from models.clip_model import CLIPMultiTaskClassifier
from models.vilt_model import ViltClassificationModel
from models.flava_model import FlavaClassificationModel
from training.trainer import train_model
from training.evaluator import evaluate_model
from training.vilt_training import train_vilt
from evaluate.vilt_evaluate import evaluate_vilt
from training.flava_training import train_flava
from evaluate.flava_evaluate import evaluate_flava
from flava_tuning import tune_hyperparameters
from dataloaders.flava_dataloader import get_flava_dataloader
from transformers import CLIPProcessor, ViltProcessor, ViltModel, FlavaProcessor, FlavaModel
from torch.utils.data import DataLoader
from utils.folder_manager import ExperimentFolderManager
from utils.helpers import save_metrics, filter_invalid_rows
from utils.logger import setup_logger
from utils.plotting import Plotting
from dataloaders.vilt_dataloader import get_vilt_dataloader
from dataloaders.clip_dataloader import get_clip_dataloader
from sklearn.preprocessing import StandardScaler
from PIL import Image
import requests
from io import BytesIO
import pytesseract
from torch.utils.data._utils.collate import default_collate
from training.lr_finder import lr_finder

import logging
experiment_logger = logging.getLogger("experiment_manager")
experiment_logger.setLevel(logging.INFO)
from utils.plotting_manager import PlottingManager

# Setting up main logger

if not experiment_logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    experiment_logger.addHandler(h)

def run_experiment(args):
    """
    Runs the experiment based on the model specified in the arguments.
    """
    global experiment_logger 
    # Use the centralized folder manager
    folder_manager = ExperimentFolderManager(base_dir=args.output_dir, model_name=args.model)
    args.log_dir = folder_manager.logs_dir
    args.output_dir = folder_manager.base_dir
    #config.log_dir = folder_manager.logs_dir
    experiment_logger = setup_logger(log_name="experiment_manager_log", log_dir=args.log_dir, sampled=args.sample)

    experiment_logger.info(f"Starting experiment for {args.model.upper()}...")


    # # Prepare folders for logs and outputs
    # model_folder = args.model.upper()
    # log_dir = os.path.join("logs", model_folder)
    # os.makedirs(log_dir, exist_ok=True)
    # output_dir = os.path.join(args.output_dir, model_folder)
    # os.makedirs(output_dir, exist_ok=True)

    # Preprocessing
    experiment_logger.info("Loading and preprocessing dataset...")
    # df = compute_pixel_values(
    #     handle_missing_values(
    #         normalize_metadata(
    #             pd.read_csv(args.data_path, sep="\t"),
    #             args.metadata_columns
    #         ),
    #         ["clean_title", "image_url", "2_way_label", "3_way_label"]
    #     ),
    #     image_column="image_url"
    # )
    # train_df, val_df, test_df = split_dataset(df)
    # Choose between loading a full batch or sampling rows
    if args.sample:
        train_df, val_df, test_df = load_batch_data(args)
        experiment_logger.info("Using the first batch from each dataset for debugging.")
    else:
    # Load preprocessed data (or preprocess if necessary)
        train_df, val_df, test_df = load_data(args)
    
    # def preprocess_dataframe(dataframe):
    #     valid_rows = []
    #     for idx, row in dataframe.iterrows():
    #         image = row.get('image_url', None)
    #         if pd.isna(image) or not is_valid_url(image):
    #             experiment_logger.warning(f"Invalid or missing image URL in row {idx}: {row}")
    #             continue
    #         valid_rows.append(row)
    #     return pd.DataFrame(valid_rows)

    # def is_valid_url(url):
    #     try:
    #         response = requests.head(url, timeout=5)
    #         return response.status_code == 200
    #     except requests.RequestException:
    #         return False

    # # Preprocess the DataFrame
    # train_df = preprocess_dataframe(train_df)
    # val_df = preprocess_dataframe(val_df)
    # test_df = preprocess_dataframe(test_df)


    # Check if we only want to preprocess and inspect the datasets
    if args.preprocess_only:
        print("Preprocessing completed. Inspecting datasets:")
        print("\nTrain Dataset Head:")
        print(train_df.head())
        print("\nValidation Dataset Head:")
        print(val_df.head())
        print("\nTest Dataset Head:")
        print(test_df.head())
        sys.exit(0)

    # Initialize plotter
    #plotter = Plotting(output_dir=args.output_dir)
    if args.model == "clip":
        #run_clip_experiment(args, train_df, val_df, test_df) 
        # CLIP-specific processing
        for include_metadata in [True, False]:  
            args.include_metadata = include_metadata
            metadata_tag = "with_metadata" if include_metadata else "without_metadata"
            experiment_logger.info(f"Starting CLIP experiment: {metadata_tag}")
            #Setup
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            train_loader = prepare_dataloader(train_df, processor, args.metadata_columns, args.batch_size, shuffle=True, include_metadata=args.include_metadata)
            val_loader = prepare_dataloader(val_df, processor, args.metadata_columns, args.batch_size,shuffle=True, include_metadata=args.include_metadata)
            test_loader = prepare_dataloader(test_df, processor, args.metadata_columns, args.batch_size,shuffle=True, include_metadata=args.include_metadata)
            
            # Initialize CLIP model
            model = CLIPMultiTaskClassifier(
                input_dim=args.input_dim,
                num_classes_2=args.num_classes_2,
                num_classes_3=args.num_classes_3,
                metadata_dim=len(args.metadata_columns)
            ).to(args.device)

            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
            plotter = PlottingManager(model_name="CLIP", folder_manager=folder_manager)

            # Initialize training history dictionary with keys for each task
            training_history = {
    "epoch": [],
    "train_loss": [],
    
    "train_accuracy_2way": [],
    "train_f1_2way": [],
    "train_precision_2way": [],
    "train_recall_2way": [],
    "train_kappa_2way": [],
    
    "train_accuracy_3way": [],
    "train_f1_3way": [],
    "train_precision_3way": [],
    "train_recall_3way": [],
    "train_kappa_3way": [],
    
    "val_loss_2way": [],
    "val_accuracy_2way": [],
    "val_f1_2way": [],
    "val_precision_2way": [],
    "val_recall_2way": [],
    "val_kappa_2way": [],
    
    "val_loss_3way": [],
    "val_accuracy_3way": [],
    "val_f1_3way": [],
    "val_precision_3way": [],
    "val_recall_3way": [],
    "val_kappa_3way": []
}



            best_val_loss = float("inf")
            patience = args.patience
            epochs_no_improve = 0

            for epoch in range(args.epochs):
                experiment_logger.info(f"Starting Epoch {epoch + 1}/{args.epochs} for {metadata_tag}")
                
                # Run training for one epoch
             #   epoch_loss = train_model(model, train_loader, optimizer, args.device, args)
                epoch_loss, metrics_2way, metrics_3way = train_model(model, train_loader, optimizer, args.device)

                # Log train metrics
               # Log train metrics
                training_history["epoch"].append(epoch + 1)
                training_history["train_loss"].append(epoch_loss)

                training_history["train_accuracy_2way"].append(metrics_2way["accuracy"])
                training_history["train_f1_2way"].append(metrics_2way["f1"])
                training_history["train_precision_2way"].append(metrics_2way["precision"])
                training_history["train_recall_2way"].append(metrics_2way["recall"])
                training_history["train_kappa_2way"].append(metrics_2way["kappa"])

                training_history["train_accuracy_3way"].append(metrics_3way["accuracy"])
                training_history["train_f1_3way"].append(metrics_3way["f1"])
                training_history["train_precision_3way"].append(metrics_3way["precision"])
                training_history["train_recall_3way"].append(metrics_3way["recall"])
                training_history["train_kappa_3way"].append(metrics_3way["kappa"])


                val_loss, val_metrics, val_labels_2, val_preds_2, val_scores_2, val_labels_3, val_preds_3, val_scores_3, _ = evaluate_model(model, val_loader, args.device)

                # --- 2-Way ---
                training_history["val_loss_2way"].append(val_metrics["2_way"]["loss"])
                training_history["val_accuracy_2way"].append(val_metrics["2_way"]["accuracy"])
                training_history["val_f1_2way"].append(val_metrics["2_way"]["f1"])
                training_history["val_precision_2way"].append(val_metrics["2_way"]["precision"])
                training_history["val_recall_2way"].append(val_metrics["2_way"]["recall"])
                training_history["val_kappa_2way"].append(val_metrics["2_way"]["kappa"])

                # --- 3-Way ---
                training_history["val_loss_3way"].append(val_metrics["3_way"]["loss"])
                training_history["val_accuracy_3way"].append(val_metrics["3_way"]["accuracy"])
                training_history["val_f1_3way"].append(val_metrics["3_way"]["f1"])
                training_history["val_precision_3way"].append(val_metrics["3_way"]["precision"])
                training_history["val_recall_3way"].append(val_metrics["3_way"]["recall"])
                training_history["val_kappa_3way"].append(val_metrics["3_way"]["kappa"])

                experiment_logger.info(f"[2-way] Accuracy: {metrics_2way['accuracy']:.4f}, F1: {metrics_2way['f1']:.4f}, Precision: {metrics_2way['precision']:.4f}, Recall: {metrics_2way['recall']:.4f}, Kappa: {metrics_2way['kappa']:.4f}")
                experiment_logger.info(f"[3-way] Accuracy: {metrics_3way['accuracy']:.4f}, F1: {metrics_3way['f1']:.4f}, Precision: {metrics_3way['precision']:.4f}, Recall: {metrics_3way['recall']:.4f}, Kappa: {metrics_3way['kappa']:.4f}")
                
                experiment_logger.info(f"[Val 2-way] Accuracy: {val_metrics['2_way']['accuracy']:.4f}, F1: {val_metrics['2_way']['f1']:.4f}, Precision: {val_metrics['2_way']['precision']:.4f}, Recall: {val_metrics['2_way']['recall']:.4f}, Kappa: {val_metrics['2_way']['kappa']:.4f}")
                experiment_logger.info(f"[Val 3-way] Accuracy: {val_metrics['3_way']['accuracy']:.4f}, F1: {val_metrics['3_way']['f1']:.4f}, Precision: {val_metrics['3_way']['precision']:.4f}, Recall: {val_metrics['3_way']['recall']:.4f}, Kappa: {val_metrics['3_way']['kappa']:.4f}")

                #experiment_logger.info(f"[{metadata_tag}] Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss (2-way): {val_metrics['2_way']['loss']:.4f}, Acc: {val_metrics['2_way']['accuracy']:.4f}")


            #     training_history["epoch"].append(epoch + 1)
            #     training_history["train_loss"].append(epoch_loss)
                
            # # Evaluate on validation set
            #     #val_loss_overall, val_metrics, _, _, _, _, _, _ = evaluate_model(model, val_loader, args.device)
            #     # Store per-task metrics if available; otherwise, use overall loss as fallback.
            #     training_history["val_loss_2way"].append(val_metrics["2_way"].get("loss", val_loss_overall))
            #     training_history["val_metrics_2way"].append(val_metrics["2_way"])
            #     training_history["val_loss_3way"].append(val_metrics["3_way"].get("loss", val_loss_overall))
            #     training_history["val_metrics_3way"].append(val_metrics["3_way"])
                
            #     experiment_logger.info(f"Epoch {epoch + 1} complete. Train Loss: {epoch_loss:.4f}, Overall Val Loss: {val_loss_overall:.4f}")
                
                # Early stopping logic
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_path = os.path.join(folder_manager.models_dir, f"clip_best_model_{metadata_tag}.pth")
                torch.save(model.state_dict(), best_model_path)
                experiment_logger.info(f"Saved best model to {best_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    experiment_logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break

            # Save training curves
            plotter.plot_training_curves(
                train_losses=training_history["train_loss"],
                val_losses=training_history["val_loss_2way"],  # can also do 3-way separately
                train_accuracies=[0]*len(training_history["train_loss"]),  # or estimate from train set
                val_accuracies=training_history["val_accuracy_2way"],
                save_as=f"clip_training_curve_2way_{metadata_tag}.png"
            )

            plotter.plot_training_curves(
                train_losses=training_history["train_loss"],
                val_losses=training_history["val_loss_3way"],
                train_accuracies=[0]*len(training_history["train_loss"]),
                val_accuracies=training_history["val_accuracy_3way"],
                save_as=f"clip_training_curve_3way_{metadata_tag}.png"
            )

            # Save CSVs
            pd.DataFrame(training_history).to_csv(
                os.path.join(folder_manager.models_dir, f"clip_training_history_{metadata_tag}.csv"),
                index=False
            )
                # Save training history to a CSV file
            # Save training history for each task
            # history_df_2way = pd.DataFrame({
            #     "epoch": training_history["epoch"],
            #     "train_loss": training_history["train_loss"],
            #     "val_loss": training_history["val_loss_2way"],
            #     "val_metrics": training_history["val_metrics_2way"]
            # })
            # history_file_2way = os.path.join(folder_manager.models_dir, f"2_way_training_history_{metadata_tag}.csv")
            # history_df_3way = os.path.join(folder_manager.models_dir, f"3_way_training_history_{metadata_tag}.csv")

            # history_df_2way.to_csv(history_file_2way, index=False)
            # experiment_logger.info(f"2-way training history saved to {history_file_2way}")

            # history_df_3way = pd.DataFrame({
            #     "epoch": training_history["epoch"],
            #     "train_loss": training_history["train_loss"],
            #     "val_loss": training_history["val_loss_3way"],
            #     "val_metrics": training_history["val_metrics_3way"]
            # })
            # history_file_3way = os.path.join(folder_manager.models_dir, f"3_way_training_history_{metadata_tag}.csv")
            # history_df_3way.to_csv(history_file_3way, index=False)
            # experiment_logger.info(f"3-way training history saved to {history_df_3way}")

            # Save the final model
            # final_model_filename = f"clip_final_model_{metadata_tag}.pth"
            # final_model_path = os.path.join(folder_manager.models_dir, final_model_filename)
            # torch.save(model.state_dict(), final_model_path)
            # experiment_logger.info(f"Final model saved to {final_model_path}")
            # Evaluate and save metrics for CLIP
            evaluate_and_save_metrics_for_clip(model, val_loader, test_loader, args, folder_manager, test_df)

    elif args.model == "vilt":
        experiment_logger.info(f"Starting experiment with metadata inclusion set to {args.include_metadata}")
        run_vilt_experiment(args, train_df, val_df, test_df, folder_manager)  
    

    elif args.model == "flava":
        for include_metadata in [True, False]:
            args.include_metadata = include_metadata
            experiment_logger.info(f"Starting experiment with metadata inclusion set to {include_metadata}")
    
            run_flava_experiment(args, train_df, val_df, test_df, folder_manager)  

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    experiment_logger.info(f"Experiment for {args.model.upper()} complete.")
 

def load_batch_data(args):
    # Define the fixed file paths for the first batch of each dataset
    train_batch_file = "data/train_batch_1.tsv"
    val_batch_file = "data/validate_batch_1.tsv"
    test_batch_file = "data/test_batch_1.tsv"

    # Load the TSV files using the tab separator
    train_df = pd.read_csv(train_batch_file, sep="\t").sample(n=50, random_state=42)
    val_df = pd.read_csv(val_batch_file, sep="\t").sample(n=20, random_state=42)
    test_df = pd.read_csv(test_batch_file, sep="\t").sample(n=20, random_state=42)

    return train_df, val_df, test_df

def load_data(args):
    # Define the paths to your preprocessed TSV files
    preprocessed_train_path = "data/TRAIN DATA/train_preprocessed.tsv"
    preprocessed_val_path = "data/VALIDATE DATA/validate_preprocessed.tsv"
    preprocessed_test_path = "data/TEST DATA/test_preprocessed.tsv"
    
    # Check if these files exist
    if (os.path.exists(preprocessed_train_path) and 
        os.path.exists(preprocessed_val_path) and 
        os.path.exists(preprocessed_test_path)):
        print("Loading preprocessed datasets...")
        train_df = pd.read_csv(preprocessed_train_path, sep="\t", nrows=2000)
        val_df = pd.read_csv(preprocessed_val_path, sep="\t", nrows=2000)
        test_df = pd.read_csv(preprocessed_test_path, sep="\t", nrows=2000)
    else:
        print("Preprocessed files not found. Running preprocessing...")
        train_df, val_df, test_df = load_and_preprocess_datasets(
            train_path=args.train_path,
            validate_path=args.validate_path,
            test_path=args.test_path,
            metadata_columns=args.metadata_columns,
            image_column="image_url"
        )
        # Save the preprocessed datasets as TSV files
        train_df.to_csv(preprocessed_train_path, sep="\t", index=False)
        val_df.to_csv(preprocessed_val_path, sep="\t", index=False)
        test_df.to_csv(preprocessed_test_path, sep="\t", index=False)
     # ✅ Save a copy of the loaded 2000-row dataset in CSV & Excel
    save_datasets(train_df, val_df, test_df)   
    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df):
    train_csv_path = "data/train_2000.csv"
    val_csv_path = "data/val_2000.csv"
    test_csv_path = "data/test_2000.csv"

    train_xlsx_path = "data/train_2000.xlsx"
    val_xlsx_path = "data/val_2000.xlsx"
    test_xlsx_path = "data/test_2000.xlsx"

    # ✅ Save as CSV (for compatibility)
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    # ✅ Save as Excel (for report writing)
    train_df.to_excel(train_xlsx_path, index=False, sheet_name="Train_2000")
    val_df.to_excel(val_xlsx_path, index=False, sheet_name="Val_2000")
    test_df.to_excel(test_xlsx_path, index=False, sheet_name="Test_2000")

    print(f"Datasets saved successfully in CSV & Excel formats!")
 # Handle missing values

def handle_missing_values_forVilt(df):
    df.dropna(subset=['clean_title', 'image_url', '2_way_label', '3_way_label', '6_way_label', 'num_comments', 'score', 'upvote_ratio'], inplace=True)
    return df

def prepare_dataloader(dataframe, processor, metadata_columns, batch_size, shuffle=False, include_metadata=False):
    class GeneralDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, processor, metadata_columns, include_metadata):
            self.dataframe = dataframe
            self.processor = processor
            self.metadata_columns = metadata_columns
            self.include_metadata = include_metadata  # Store the flag as an attribute

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            text = row['clean_title']
            # Get the image URL from the dataframe
            image_url = row.get('image_url', None)
            if image_url is None or pd.isna(image_url):
                experiment_logger.error(f"Missing image URL in row {idx}: {row}")
                return None  # Skip this row

            # If include_metadata is True, extract metadata values
            metadata_values = (row[self.metadata_columns].values.astype(float)
                               if self.include_metadata else [])
            
            # Process the image using your preprocessing function
            image = preprocess_image(image_url)
            if image is None:
                #experiment_logger.error(f"Could not process image at URL: {image_url} in row {idx}")
                # wherever it's used
                print(f"[experiment_manager] Could not process image at URL: {image_url} in row {idx}")

                return None

            # Use the processor to calculate pixel values from the image
            # This returns a dictionary with a key like "pixel_values"
            image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            # If by chance the tensor has an extra dimension, remove it
            if image_tensor.dim() == 5:
                image_tensor = image_tensor.squeeze(0)
                experiment_logger.info(f"Image tensor shape after processing: {image_tensor.shape}")

            # Now, call the processor to generate inputs for text and image
            inputs = self.processor(
                text=[text],
                images=image,  # The processor will compute pixel values internally
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=40
            )
            # Remove any extra batch dimension from pixel_values
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
            
            # If metadata is to be included, add it to the inputs
            if self.include_metadata:
                inputs["metadata"] = torch.tensor(metadata_values, dtype=torch.float32)
            
            # (Optional) You can add label processing here if needed.
            # For now, this function focuses on text, image, and metadata.
            # Add labels for 2-way and 3-way classification.
            inputs["labels_2_way"] = torch.tensor(row["2_way_label"], dtype=torch.long)
            inputs["labels_3_way"] = torch.tensor(row["3_way_label"], dtype=torch.long)
            inputs["row_index"] = torch.tensor(idx)
            
            # Debug logging (remove or adjust if needed)
            # experiment_logger.info(
            #     f"Dataset Sample - input_ids: {inputs['input_ids'].shape}, "
            #     f"pixel_values: {inputs['pixel_values'].shape}"
            # )
           

            return {key: val.squeeze(0) if key != "row_index" else val for key, val in inputs.items()}

    return DataLoader(GeneralDataset(dataframe, processor, metadata_columns, include_metadata),
                      batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)


def custom_collate_fn(batch):
    # Filter out None samples
    valid_batch = [item for item in batch if item is not None]
    if len(valid_batch) == 0:
        return {}  # or raise an error if desired
    return default_collate(valid_batch)

# def evaluate_and_save_metrics(model, val_loader, test_loader, args):
#     """
#     Evaluate the model on validation and test sets and save metrics and plots.
#     """
#     model.eval()  # Switch model to evaluation mode

#     # Initialize PlottingManager
#     plotter = PlottingManager(model_name="CLIP", output_dir=args.output_dir)

#     # Evaluate on validation set
#     evaluate_split_and_plot(
#         model=model,
#         dataloader=val_loader,
#         device=args.device,
#         labels_2=args.labels_2,
#         labels_3=args.labels_3,
#         split_name="validation",
#         plotter=plotter
#     )

#     # Evaluate on test set
#     evaluate_split_and_plot(
#         model=model,
#         dataloader=test_loader,
#         device=args.device,
#         labels_2=args.labels_2,
#         labels_3=args.labels_3,
#         split_name="test",
#         plotter=plotter
#     )

# def evaluate_split_and_plot(model, dataloader, device, labels_2, labels_3, split_name, plotter):
#     """
#     Evaluate a specific dataset split and generate corresponding plots.
#     """
#     experiment_logger.info(f"Evaluating on {split_name} set...")

#     # Evaluate the model and extract metrics
#     loss, metrics, labels_2_true, preds_2, scores_2, labels_3_true, preds_3, scores_3 = evaluate_model(
#         model=model,
#         dataloader=dataloader,
#         device=device
#     )

#     # Save metrics
#     metrics_filename = f"{split_name}_metrics.json"
#     plotter.save_metrics(metrics, filename=metrics_filename)
#     experiment_logger.info(f"{split_name.capitalize()} Metrics: {metrics}")

#     # Generate plots for 2-way classification
#     plotter.plot_confusion_matrix(
#         labels_2_true, preds_2, labels=labels_2,
#         save_as=f"{split_name}_2_way_confusion_matrix.png"
#     )
#     plotter.plot_roc_curve(
#         labels_2_true, [s[1] for s in scores_2],  # Extract positive class probabilities
#         save_as=f"{split_name}_2_way_roc_curve.png",
#         labels=labels_2
#     )
#     plotter.plot_precision_recall_curve(
#         labels_2_true, [s[1] for s in scores_2],
#         save_as=f"{split_name}_2_way_precision_recall_curve.png",
#         labels=labels_2
#     )

#     # Generate plots for 3-way classification
#     plotter.plot_confusion_matrix(
#         labels_3_true, preds_3, labels=labels_3,
#         save_as=f"{split_name}_3_way_confusion_matrix.png"
#     )
#     plotter.plot_roc_curve(
#         labels_3_true, scores_3,
#         save_as=f"{split_name}_3_way_roc_curve.png",
#         labels=labels_3
#     )
#     plotter.plot_precision_recall_curve(
#         labels_3_true, scores_3,
#         save_as=f"{split_name}_3_way_precision_recall_curve.png",
#         labels=labels_3
#     )
def evaluate_and_save_metrics_for_clip(model, val_loader, test_loader, args, folder_manager, test_df=None):
    """
    Evaluate the CLIP model and save plots, metrics, and detailed predictions.
    """
    metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"
    model.eval()

    plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)

    # --- Validation Evaluation ---
    experiment_logger.info("🔎 Evaluating on validation set...")
    val_loss, val_metrics, val_labels_2, val_preds_2, val_scores_2, val_labels_3, val_preds_3, val_scores_3, _ = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=args.device
    )

    # Save validation metrics
    val_metrics_file = f"validation_metrics_{args.model}_{metadata_tag}.json"
    plotter.save_metrics(val_metrics, filename=val_metrics_file)
    experiment_logger.info(f"✅ Saved validation metrics: {val_metrics_file}")

    # Plots: 2-way
    plotter.plot_confusion_matrix(val_labels_2, val_preds_2, labels=args.labels_2, save_as=f"validation_2_way_confusion_matrix_{metadata_tag}.png")
    plotter.plot_roc_curve(val_labels_2, val_scores_2, save_as=f"validation_2_way_roc_curve_{metadata_tag}.png", labels=args.labels_2)
    plotter.plot_precision_recall_curve(val_labels_2, val_scores_2, save_as=f"validation_2_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_2)

    # Plots: 3-way
    plotter.plot_confusion_matrix(val_labels_3, val_preds_3, labels=args.labels_3, save_as=f"validation_3_way_confusion_matrix_{metadata_tag}.png")
    plotter.plot_roc_curve(val_labels_3, val_scores_3, save_as=f"validation_3_way_roc_curve_{metadata_tag}.png", labels=args.labels_3)
    plotter.plot_precision_recall_curve(val_labels_3, val_scores_3, save_as=f"validation_3_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_3)

    # --- Test Evaluation ---
    experiment_logger.info("🔎 Evaluating on test set...")
    test_loss, test_metrics, test_labels_2, test_preds_2, test_scores_2, test_labels_3, test_preds_3, test_scores_3, test_row_indices = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=args.device
    )

    # Save test metrics
    test_metrics_file = f"test_metrics_{args.model}_{metadata_tag}.json"
    plotter.save_metrics(test_metrics, filename=test_metrics_file)
    experiment_logger.info(f"Saved test metrics: {test_metrics_file}")

    # Test plots - 2-way
    plotter.plot_confusion_matrix(test_labels_2, test_preds_2, labels=args.labels_2, save_as=f"test_2_way_confusion_matrix_{metadata_tag}.png")
    plotter.plot_roc_curve(test_labels_2, test_scores_2, save_as=f"test_2_way_roc_curve_{metadata_tag}.png", labels=args.labels_2)
    plotter.plot_precision_recall_curve(test_labels_2, test_scores_2, save_as=f"test_2_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_2)

    # Test plots - 3-way
    plotter.plot_confusion_matrix(test_labels_3, test_preds_3, labels=args.labels_3, save_as=f"test_3_way_confusion_matrix_{metadata_tag}.png")
    plotter.plot_roc_curve(test_labels_3, test_scores_3, save_as=f"test_3_way_roc_curve_{metadata_tag}.png", labels=args.labels_3)
    plotter.plot_precision_recall_curve(test_labels_3, test_scores_3, save_as=f"test_3_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_3)

    # --- Save detailed prediction CSV (if test_df is available) ---
    if test_df is not None:
        used_test_df = test_df.iloc[test_row_indices].reset_index(drop=True)

        prediction_df = pd.DataFrame({
            "true_label_2": test_labels_2,
            "predicted_label_2": test_preds_2,
            "prob_class_0_2": [p[0] for p in test_scores_2],
            "prob_class_1_2": [p[1] for p in test_scores_2],
            "true_label_3": test_labels_3,
            "predicted_label_3": test_preds_3,
        })

        for i in range(len(test_scores_3[0])):
            prediction_df[f"prob_class_{i}_3"] = [p[i] for p in test_scores_3]

        final_df = pd.concat([used_test_df.reset_index(drop=True), prediction_df], axis=1)

        pred_csv_path = os.path.join(folder_manager.metrics_dir, f"clip_predictions_detailed_{metadata_tag}.csv")
        final_df.to_csv(pred_csv_path, index=False)
        experiment_logger.info(f"Saved test predictions CSV: {pred_csv_path} with {final_df.shape[0]} rows")
        # Also save Excel version
        pred_xlsx_path = pred_csv_path.replace(".csv", ".xlsx")
        final_df.to_excel(pred_xlsx_path, index=False)
        experiment_logger.info(f"Saved test predictions Excel: {pred_xlsx_path}")

    else:
        experiment_logger.warning("test_df not provided — skipping saving detailed prediction CSV.")

# def evaluate_and_save_metrics(model, val_loader, test_loader, args, folder_manager):
#     """
#     Evaluate the model on validation and test sets, and save metrics.
#     """
#     # Create a unique tag for this run:
#     metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"
#     model.eval()  # Ensure the model is in evaluation mode

#     #plotter = Plotting(output_dir=args.output_dir)
#     plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)

#     # Evaluate on the validation 
    
#     experiment_logger.info("Evaluating on validation set...")
#     val_loss, val_metrics, val_labels_2, val_preds_2, val_scores_2, val_labels_3, val_preds_3, val_scores_3, row_indices = evaluate_model(
#         model=model,
#         dataloader=val_loader,
#         device=args.device
#     )
#     # Save validation metrics using the Plotting manager
#     val_metrics_filename = f"validation_metrics_{args.model}_{metadata_tag}.json"
#     plotter.save_metrics(val_metrics, filename=val_metrics_filename)
#     experiment_logger.info(f"Validation Metrics: {val_metrics}")

#     # Generate validation plots
#     plotter.plot_confusion_matrix(val_labels_2, val_preds_2, labels=args.labels_2, save_as=f"validation_2_way_confusion_matrix_{metadata_tag}.png")
#     plotter.plot_roc_curve(val_labels_2, val_scores_2, save_as=f"validation_2_way_roc_curve_{metadata_tag}.png", labels=args.labels_2)
#     plotter.plot_precision_recall_curve(val_labels_2, val_scores_2, save_as=f"validation_2_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_2)

#     plotter.plot_confusion_matrix(val_labels_3, val_preds_3, labels=args.labels_3, save_as=f"validation_3_way_confusion_matrix_{metadata_tag}.png")
#     plotter.plot_roc_curve(val_labels_3, val_scores_3, save_as=f"validation_3_way_roc_curve_{metadata_tag}.png", labels=args.labels_3)
#     plotter.plot_precision_recall_curve(val_labels_3, val_scores_3, save_as=f"validation_3_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_3)

#     # Evaluate on the test set
#     experiment_logger.info("Evaluating on test set...")
#     test_loss, test_metrics, test_labels_2, test_preds_2, test_scores_2, test_labels_3, test_preds_3, test_scores_3 = evaluate_model(
#         model=model,
#         dataloader=test_loader,
#         device=args.device
#     )
#     # Save test metrics using the Plotting manager
#     test_metrics_filename = f"test_metrics_{args.model}_{metadata_tag}.json"
#     plotter.save_metrics(test_metrics, filename=test_metrics_filename)
#     experiment_logger.info(f"Test Metrics: {test_metrics}")

#     # Generate test plots
#     plotter.plot_confusion_matrix(test_labels_2, test_preds_2, labels=args.labels_2, save_as=f"test_2_way_confusion_matrix_{metadata_tag}.png")
#     plotter.plot_roc_curve(test_labels_2, test_scores_2, save_as=f"test_2_way_roc_curve_{metadata_tag}.png", labels=args.labels_2)
#     plotter.plot_precision_recall_curve(test_labels_2, test_scores_2, save_as=f"test_2_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_2)

#     plotter.plot_confusion_matrix(test_labels_3, test_preds_3, labels=args.labels_3, save_as=f"test_3_way_confusion_matrix_{metadata_tag}.png")
#     plotter.plot_roc_curve(test_labels_3, test_scores_3, save_as=f"test_3_way_roc_curve_{metadata_tag}.png", labels=args.labels_3)
#     plotter.plot_precision_recall_curve(test_labels_3, test_scores_3, save_as=f"test_3_way_precision_recall_curve_{metadata_tag}.png", labels=args.labels_3)

def evaluate_and_save_metrics_for_flava(
    model, val_loader_2way, val_loader_3way, test_loader_2way, test_loader_3way, args,folder_manager, test_df
):
    """
    Evaluate the model on validation and test sets, and save metrics and plots.
    """
    model.eval()  # Ensure the model is in evaluation mode
    plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)

    # Create a tag based on whether metadata is included
    metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"

    tasks = []
    if val_loader_2way and test_loader_2way:
        tasks.append({
            "name": "2-way classification",
            "val_loader": val_loader_2way,
            "test_loader": test_loader_2way,
            "labels": args.labels_2,
        })
    if val_loader_3way and test_loader_3way:
        tasks.append({
            "name": "3-way classification",
            "val_loader": val_loader_3way,
            "test_loader": test_loader_3way,
            "labels": args.labels_3,
        })

    for task in tasks:
        task_name = task["name"]
        val_loader = task["val_loader"]
        test_loader = task["test_loader"]
        labels = task["labels"]

        experiment_logger.info(f"Evaluating {task_name}...")

        # Validation Evaluation (use cached outputs if available)
        val_metrics, (val_labels, val_preds, val_scores, _, _, test_row_indices) = evaluate_flava(model, val_loader, args.device)

        # Save validation metrics with metadata tag in the filename
        val_metrics_filename = f"{task_name.replace(' ', '_')}_{metadata_tag}_validation_metrics.json"
        plotter.save_metrics(val_metrics, filename=val_metrics_filename)
        experiment_logger.info(f"{task_name} Validation Metrics: {val_metrics}")

        # Validation Plots
        plotter.plot_confusion_matrix(
            val_labels, val_preds, labels=labels, save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_validation_confusion_matrix.png"
        )
        if len(labels) == 2:
            plotter.plot_roc_curve(
                val_labels, val_scores[:, 1], save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_validation_roc_curve.png"
            )
            plotter.plot_precision_recall_curve(
                val_labels, val_scores[:, 1], save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_validation_precision_recall_curve.png"
            )
        else:
            plotter.plot_roc_curve(
                val_labels, val_scores, save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_validation_multiclass_roc_curve.png", labels=labels
            )
            plotter.plot_precision_recall_curve(
                val_labels, val_scores, save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_validation_multiclass_precision_recall_curve.png", labels=labels
            )

        # Test Evaluation
        experiment_logger.info(f"Evaluating on {task_name} test set...")
        test_metrics, (test_labels, test_preds, test_scores, _, _,test_row_indices) = evaluate_flava(
            model=model, dataloader=test_loader, device=args.device
        )
        # ✅ Map back to original test dataframe
        used_test_df = test_df.iloc[test_row_indices].reset_index(drop=True)

        # ✅ Save predictions + probabilities
        results_df = pd.DataFrame({
            "true_label": test_labels,
            "predicted_label": test_preds
        })

        probs_df = pd.DataFrame(test_scores, columns=[f"prob_class_{i}" for i in range(len(test_scores[0]))])
        final_df = pd.concat([used_test_df.reset_index(drop=True), results_df, probs_df], axis=1)

        # ✅ Save CSV
        csv_save_path = os.path.join(folder_manager.metrics_dir, f"{task_name.replace(' ', '_')}_{metadata_tag}_test_predictions_detailed.csv")
        final_df.to_csv(csv_save_path, index=False)
        experiment_logger.info(f"📝 Saved detailed predictions to {csv_save_path}")

        test_metrics_filename = f"{task_name.replace(' ', '_')}_{metadata_tag}_test_metrics.json"
        plotter.save_metrics(test_metrics, filename=test_metrics_filename)
        experiment_logger.info(f"{task_name} Test Metrics: {test_metrics}")

        # Test Plots
        plotter.plot_confusion_matrix(
            test_labels, test_preds, labels=labels, save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_test_confusion_matrix.png"
        )
        if len(labels) == 2:
            plotter.plot_roc_curve(
                test_labels, test_scores[:, 1], save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_test_roc_curve.png"
            )
            plotter.plot_precision_recall_curve(
                test_labels, test_scores[:, 1], save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_test_precision_recall_curve.png"
            )
        else:
            plotter.plot_roc_curve(
                test_labels, test_scores, save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_test_multiclass_roc_curve.png", labels=labels
            )
            plotter.plot_precision_recall_curve(
                test_labels, test_scores, save_as=f"{task_name.replace(' ', '_')}_{metadata_tag}_test_multiclass_precision_recall_curve.png", labels=labels
            )

    experiment_logger.info("Evaluation complete for all tasks.")


def convert_to_serializable(metrics):
    # Convert tensors or numpy arrays to lists or floats
    return {key: (value.tolist() if isinstance(value, torch.Tensor) else value) for key, value in metrics.items()}


def run_flava_experiment(args, train_df, val_df, test_df, folder_manager):
    # Initialize the base FLAVA model 
    # STEP1 train batch 1 and then whole dataset shoot!!
    # RUN WITH METADATA AND ALSO WITHOUT METADATA 
    # SAVE MODELS OF EACH w and wo Metadata (2 way and 3 way)
    # STEP1 test batch 1 and then whole dataset shoot!! (graphs etc)
    # MAKE SURE TUNING KAB KAHA HO RAHA 
    experiment_logger.info(f"Metadata inclusion: {'Enabled' if args.include_metadata else 'Disabled'}")
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    flava_model = FlavaModel.from_pretrained("facebook/flava-full")
    model = FlavaClassificationModel(flava_model, num_labels=2, metadata_dim=len(args.metadata_columns) if args.include_metadata else 0,
    include_metadata=args.include_metadata).to(args.device) 
    experiment_logger.info(f"Model initialized with metadata_dim={model.metadata_dim}, include_metadata={model.include_metadata}") # Default to 2-way

    # Fine-tune the model or only the classifier
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True  # Always fine-tune the classifier
        else:
            param.requires_grad = args.fine_tune  # Controlled by the `--fine_tune` argument

    # # Define the optimizer
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),  # Only include trainable parameters
    #     lr=args.learning_rate
    # )


    # # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # Example scheduler

    # # Early stopping parameters
    # patience = args.patience
    # best_val_loss = float("inf")
    # epochs_without_improvement = 0

    # Before creating dataloaders
    # Log the full sampled DataFrames
    # experiment_logger.info("Logging entire train_df sample:")
    # experiment_logger.info(f"\n{train_df}")

    # experiment_logger.info("Logging entire val_df sample:")
    # experiment_logger.info(f"\n{val_df}")

    # experiment_logger.info("Logging entire test_df sample:")
    # experiment_logger.info(f"\n{test_df}")

    # # Log label counts to ensure they're correctly distributed
    # experiment_logger.info(f"Train 2-way label counts:\n{train_df['2_way_label'].value_counts()}")
    # experiment_logger.info(f"Train 3-way label counts:\n{train_df['3_way_label'].value_counts()}")

    # experiment_logger.info(f"Validation 2-way label counts:\n{val_df['2_way_label'].value_counts()}")
    # experiment_logger.info(f"Validation 3-way label counts:\n{val_df['3_way_label'].value_counts()}")

    # experiment_logger.info(f"Test 2-way label counts:\n{test_df['2_way_label'].value_counts()}")
    # experiment_logger.info(f"Test 3-way label counts:\n{test_df['3_way_label'].value_counts()}")



    # Prepare data loaders for 2-way and 3-way tasks
    train_loader_2way = get_flava_dataloader(train_df, processor, "2_way_label", args.batch_size, args.include_metadata, args.metadata_columns if args.include_metadata else None)
    val_loader_2way = get_flava_dataloader(val_df, processor, "2_way_label", args.batch_size, args.include_metadata, args.metadata_columns if args.include_metadata else None)
    test_loader_2way = get_flava_dataloader(test_df, processor, "2_way_label", args.batch_size, args.include_metadata, args.metadata_columns if args.include_metadata else None)

    train_loader_3way = get_flava_dataloader(train_df, processor, "3_way_label", args.batch_size, args.include_metadata, args.metadata_columns if args.include_metadata else None)
    val_loader_3way = get_flava_dataloader(val_df, processor, "3_way_label", args.batch_size, args.include_metadata, args.metadata_columns if args.include_metadata else None)
    test_loader_3way = get_flava_dataloader(test_df, processor, "3_way_label", args.batch_size, args.include_metadata, args.metadata_columns if args.include_metadata else None)


    # Explicitly handle 2-way classification
    experiment_logger.info("Starting 2-way classification...")
    model.update_num_labels(num_labels=2)
    perform_flava_task(
        model=model,
        train_loader=train_loader_2way,
        val_loader=val_loader_2way,
        test_loader=test_loader_2way,
        task_name="2-way classification",
        args=args,
        folder_manager=folder_manager,
        test_df=test_df

    )

    # Explicitly handle 3-way classification
    experiment_logger.info("Starting 3-way classification...")
    model.update_num_labels(num_labels=3)
    perform_flava_task(
        model=model,
        train_loader=train_loader_3way,
        val_loader=val_loader_3way,
        test_loader=test_loader_3way,
        task_name="3-way classification",
        args=args,
        folder_manager=folder_manager,
        test_df= test_df
    )
    # Training and evaluation tasks
    # tasks = [
    #     {
    #         "name": "2-way classification",
    #         "num_labels": 2,
    #         "train_loader": train_loader_2way,
    #         "val_loader": val_loader_2way,
    #         "test_loader": test_loader_2way,
    #     },
    #     {
    #         "name": "3-way classification",
    #         "num_labels": 3,
    #         "train_loader": train_loader_3way,
    #         "val_loader": val_loader_3way,
    #         "test_loader": test_loader_3way,
    #     },
    # ]
    #  # Aggregated metrics for analysis
    # all_metrics = []

    # for task in tasks:
    #     task_name = task["name"]
    #     num_labels = task["num_labels"]
    #     train_loader = task["train_loader"]
    #     val_loader = task["val_loader"]
    #     test_loader = task["test_loader"]

    #     #experiment_logger.info(f"Starting training for {task_name}...")
    #     experiment_logger.info(f"Starting {task_name} with {num_labels} labels.")
    #     model.update_num_labels(num_labels=num_labels)  # Update classifier for the task
    #     #optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)  # Reinitialize optimizer

    #     # Train the model
    #     for epoch in range(args.epochs):
    #         experiment_logger.info(f"{task_name} - Epoch {epoch + 1}/{args.epochs}")
            
    #         train_loss, train_losses, train_accuracies, overall_accuracy, overall_f1 = train_flava(
    #         model, train_loader, optimizer, args.device
    #         )
    #         experiment_logger.info(
    #         f"Training metrics - Loss: {train_loss}, Accuracy: {overall_accuracy}, F1-Score: {overall_f1}"
    #         )

    #         scheduler.step()  # Update learning rate with scheduler

    #         # Validation after each epoch
    #         val_loss, val_metrics, val_labels, val_preds, val_scores = evaluate_flava(
    #             model, val_loader, args.device
    #         )
    #         all_metrics.append(val_metrics)

    #         experiment_logger.info(f"Validation Metrics after Epoch {epoch + 1}: {val_metrics}")

    #         # Early stopping logic
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             epochs_without_improvement = 0
    #             # Save the best model
    #             # Determine the filename based on metadata inclusion
    #             metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"
    #             best_model_path = os.path.join(folder_manager.models_dir, f"{task_name.replace(' ', '_')}_{metadata_tag}_best_model.pth")

    #             # Save the best model
    #             torch.save(model.state_dict(), best_model_path)
    #             experiment_logger.info(f"Best model for {task_name} ({metadata_tag}) saved at {best_model_path}")
    #         else:
    #             epochs_without_improvement += 1

    #         if epochs_without_improvement >= patience:
    #             experiment_logger.info(f"Early stopping triggered for {task_name}.")
    #             break
        
    #     # Final evaluation after training
    #     experiment_logger.info(f"Evaluating {task_name}...")
    #     evaluate_and_save_metrics_for_flava(
    #         model=model,
    #         val_loader_2way=val_loader if num_labels == 2 else None,
    #         val_loader_3way=val_loader if num_labels == 3 else None,
    #         test_loader_2way=test_loader if num_labels == 2 else None,
    #         test_loader_3way=test_loader if num_labels == 3 else None,
    #         args=args,
    #         folder_manager=folder_manager,
    #         )
    #     # Save aggregated metrics
    #     aggregated_metrics_path = os.path.join(folder_manager.metrics_dir, "aggregated_metrics.json")
    #     with open(aggregated_metrics_path, "w") as f:
    #         json.dump(all_metrics, f, indent=4)
    #     experiment_logger.info(f"Aggregated metrics saved at {aggregated_metrics_path}")

    #     # Save metadata for SHAP
    #     experiment_logger.info("Saving training metadata for SHAP explanations...")
    #     training_metadata_values = train_df[args.metadata_columns].values.tolist()
    #     metadata_save_path = os.path.join(folder_manager.base_dir, "training_metadata.pkl")
    #     with open(metadata_save_path, "wb") as f:
    #         pickle.dump(training_metadata_values, f)
    #     experiment_logger.info(f"Training metadata saved at {metadata_save_path}.")

    experiment_logger.info("FLAVA experiment complete.")


        #     experiment_logger.info(
        #     f"{task_name} - Epoch {epoch + 1}/{args.epochs}, "
        #     f"Loss: {train_loss}, Accuracy: {sum(train_accuracies) / len(train_accuracies)}"
        #     )
        # # Save the model for this task
        # save_path = os.path.join(args.output_dir, f"{task_name.replace(' ', '_')}_model.pth")
        # torch.save(model.state_dict(), save_path)
        # experiment_logger.info(f"{task_name} model saved at {save_path}")

        # Evaluate the model
        


def perform_flava_task(model, train_loader, val_loader, test_loader, task_name, args, folder_manager, test_df):
    # Define optimizer and scheduler
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.learning_rate,
    # )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.5, verbose=True)

    # Early stopping parameters
    patience = args.patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0



    # Initialize training history dictionary
    training_history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_loss": [],
        "val_metrics": []
    }

    experiment_logger.info(f"Starting {task_name}...training")
    if getattr(args, "run_lr_finder", False):
        experiment_logger.info("🔍 Running LR Finder for ViLT...")
        best_lr, lrs, losses = lr_finder(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=args.device,
            #save_path=lr_plot_path  # handled by PlottingManager
    )
        # Plot results (using a PlottingManager for VILT).
        plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)

        # Save LR Finder plot
        lr_plot_path = os.path.join(folder_manager.images_dir, f"{task_name.replace(' ', '_')}_lr_finder_plot.png")
        plotter.plot_lr_finder(lrs=lrs, losses=losses, save_as=f"{task_name.replace(' ', '_')}_lr_finder_plot.png")
        experiment_logger.info(f"LR Finder plot saved to {lr_plot_path}")
        # Use best LR for training
        args.learning_rate = best_lr
        experiment_logger.info(f"Best learning rate selected from LR Finder: {best_lr:.6f}")

    else:
        experiment_logger.info("Skipping LR Finder (use --run_lr_finder to enable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    for epoch in range(args.epochs):
        experiment_logger.info(f"{task_name} - Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_losses, train_accuracies, overall_accuracy, overall_f1 = train_flava(
            model, train_loader, optimizer, args.device
        )

        # Validation val_loss, val_metrics, val_labels, val_preds, val_probs, val_indices
        val_metrics, (val_labels, val_preds, val_probs, val_losses, val_accuracies, val_indices) = evaluate_flava(
        model, val_loader, args.device)
    
        # Get val_loss after evaluation
        val_loss = val_metrics["loss"]

        experiment_logger.info(
            f"Training metrics - Loss: {train_loss}, Accuracy: {overall_accuracy}, F1-Score: {overall_f1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        
        experiment_logger.info(
            f"Training metrics - Loss: {train_loss}, Accuracy: {overall_accuracy}, F1-Score: {overall_f1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        # Update training history
        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(overall_accuracy)
        training_history["train_f1"].append(overall_f1)
        training_history["val_loss"].append(val_loss)
        training_history["val_metrics"].append(val_metrics)

        scheduler.step(val_loss)

        # Checkpointing / Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"
            best_model_path = os.path.join(
                folder_manager.models_dir,
                f"{task_name.replace(' ', '_')}_{metadata_tag}_best_model.pth",
            )
            torch.save(model.state_dict(), best_model_path)
            experiment_logger.info(f"Best model for {task_name} ({metadata_tag}) saved at {best_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            experiment_logger.info(f"Early stopping triggered for {task_name}.")
            break
    # Save training history to a CSV file
    history_df = pd.DataFrame(training_history)
    history_file = os.path.join(folder_manager.models_dir, f"{task_name.replace(' ', '_')}_training_history_{metadata_tag}.csv")
    history_df.to_csv(history_file, index=False)
    experiment_logger.info(f"Training history saved to {history_file}")

        # Plot training/validation loss and accuracy curves
    plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)
    plotter.plot_training_curves(
        train_losses=training_history["train_loss"],
        val_losses=training_history["val_loss"],
        train_accuracies=training_history["train_accuracy"],
        val_accuracies=[m["accuracy"] for m in training_history["val_metrics"]],
        save_as=f"{task_name.replace(' ', '_')}_training_validation_curve_{metadata_tag}.png"
    )
    experiment_logger.info(f"Training curves saved for {task_name} ({metadata_tag})")


    # Final evaluation
    experiment_logger.info(f"Evaluating {task_name}...")
    evaluate_and_save_metrics_for_flava(
        model=model,
        val_loader_2way=val_loader if task_name == "2-way classification" else None,
        val_loader_3way=val_loader if task_name == "3-way classification" else None,
        test_loader_2way=test_loader if task_name == "2-way classification" else None,
        test_loader_3way=test_loader if task_name == "3-way classification" else None,
        args=args,
        folder_manager=folder_manager,
        test_df= test_df
    )


def run_vilt_experiment(args, train_df, val_df, test_df, folder_manager):
    experiment_logger.info(f"Metadata inclusion: {'Enabled' if args.include_metadata else 'Disabled'}")

    # Loop over both metadata configurations: True and False
    for include_meta in [True, False]:
        args.include_metadata = include_meta
        metadata_tag = "with_metadata" if include_meta else "without_metadata"
        experiment_logger.info(f"Starting ViLT experiment: {metadata_tag}")
    
        experiment_logger.info("Preparing data loaders for ViLT...")
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

        # Prepare data loaders for 2-way and 3-way classification
        train_loader_2way, train_class_weights_2way = get_vilt_dataloader(
            train_df, processor, label_type='2_way_label', batch_size=args.batch_size, shuffle=True, metadata_columns=args.metadata_columns
        )
        val_loader_2way, _ = get_vilt_dataloader(
            val_df, processor, label_type='2_way_label', batch_size=args.batch_size, shuffle=False, metadata_columns=args.metadata_columns
        )
        test_loader_2way, _ = get_vilt_dataloader(
            test_df, processor, label_type='2_way_label', batch_size=args.batch_size, shuffle=False, metadata_columns=args.metadata_columns
        )

        train_loader_3way, train_class_weights_3way = get_vilt_dataloader(
            train_df, processor, label_type='3_way_label', batch_size=args.batch_size, shuffle=True, metadata_columns=args.metadata_columns
        )
        val_loader_3way, _ = get_vilt_dataloader(
            val_df, processor, label_type='3_way_label', batch_size=args.batch_size, shuffle=False, metadata_columns=args.metadata_columns
        )
        test_loader_3way, _ = get_vilt_dataloader(
            test_df, processor, label_type='3_way_label', batch_size=args.batch_size, shuffle=False, metadata_columns=args.metadata_columns
        )

        experiment_logger.info("ViLT data loaders prepared.")

        tasks = [
            {
                "name": "2-way classification",
                "num_labels": args.num_classes_2,
                "train_loader": train_loader_2way,
                "val_loader": val_loader_2way,
                "test_loader": test_loader_2way,
                "class_weights": train_class_weights_2way,
                "labels": args.labels_2,
            },
            {
                "name": "3-way classification",
                "num_labels": args.num_classes_3,
                "train_loader": train_loader_3way,
                "val_loader": val_loader_3way,
                "test_loader": test_loader_3way,
                "class_weights": train_class_weights_3way,
                "labels": args.labels_3,
            },
        ]

        for task in tasks:
            task_name = task["name"]
            num_labels = task["num_labels"]
            train_loader = task["train_loader"]
            val_loader = task["val_loader"]
            test_loader = task["test_loader"]
            class_weights = task["class_weights"]
            labels = task["labels"]

            experiment_logger.info(f"Starting {task_name} with {num_labels} labels.")

            # Reinitialize the model for the current task
            vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
            model = ViltClassificationModel(
                vilt_model=vilt_model,
                num_labels=num_labels,
                metadata_dim=len(args.metadata_columns),
                include_metadata=args.include_metadata
            ).to(args.device)
            perform_vilt_task(model, task["train_loader"], task["val_loader"], task["test_loader"], test_df, task_name,num_labels, labels, args, folder_manager)
            
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        #         # Use an adaptive scheduler for example.
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
                
        #         # Initialize training history for the current task.
        #     training_history = {
        #         "epoch": [],
        #         "train_loss": [],
        #         "val_loss": [],
        #         "val_metrics": []
        #     }
                
        #     # Train
        # for epoch in range(args.epochs):
        #     experiment_logger.info(f"Training {task_name}..{metadata_tag}..")
        #     # Train for one epoch.
        #     epoch_loss = train_vilt(model, train_loader, optimizer, args.device, class_weights)

        #     training_history["epoch"].append(epoch + 1)
        #     training_history["train_loss"].append(epoch_loss)

        #     # Evaluate on the validation set.
        #     val_loss, val_metrics, val_labels, val_preds, val_probs = evaluate_vilt(model, val_loader, args.device)
        #     training_history["val_loss"].append(val_loss)
        #     training_history["val_metrics"].append(val_metrics)
        #     if isinstance(val_loss, tuple):
        #         val_loss = val_loss[0]  # Extract the float from the tuple
        #     else:
        #         val_loss = float(val_loss)
        #     #experiment_logger.info(f"Task: {task_name} | Epoch {epoch+1} complete. Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        #     scheduler.step(val_loss)

        # # Save training history.
        # history_df = pd.DataFrame(training_history)
        # history_filename = f"{task_name.replace(' ', '_')}_training_history_{metadata_tag}.csv"
        # history_file = os.path.join(folder_manager.models_dir, history_filename)
        # history_df.to_csv(history_file, index=False)
        # experiment_logger.info(f"Training history for {task_name} saved to {history_file}")

        # # Save the final model.
        # final_model_filename = f"vilt_final_model_{task_name.replace(' ', '_')}_{metadata_tag}.pth"
        # final_model_path = os.path.join(folder_manager.models_dir, final_model_filename)
        # torch.save(model.state_dict(), final_model_path)
        # experiment_logger.info(f"Final model for {task_name} saved to {final_model_path}")

        # # Evaluate on test set and save metrics/plots.
        # experiment_logger.info(f"Evaluating {task_name} on validation set...")
        # val_metrics, val_labels, val_preds, val_probs = evaluate_vilt(model, val_loader, args.device)
        # experiment_logger.info(f"Evaluating {task_name} on test set...")
        # test_metrics, test_labels, test_preds, test_probs = evaluate_vilt(model, test_loader, args.device)

        # # Save metrics with filenames that include task name and metadata tag.
        # val_metrics_filename = f"{task_name.replace(' ', '_')}_val_metrics_{metadata_tag}.json"
        # save_metrics(val_metrics, folder_manager.metrics_dir, val_metrics_filename)
        # test_metrics_filename = f"{task_name.replace(' ', '_')}_test_metrics_{metadata_tag}.json"
        # save_metrics(test_metrics, folder_manager.metrics_dir, test_metrics_filename)
        # experiment_logger.info(f"Metrics saved for {task_name} (Validation: {val_metrics_filename}, Test: {test_metrics_filename})")

        # # Plot results (using a PlottingManager for VILT).
        # plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)
        # plotter.plot_confusion_matrix(y_true=test_labels, y_pred=test_preds, labels=labels,
        #                                     save_as=f"{task_name.replace(' ', '_')}_confusion_matrix_{metadata_tag}.png")
        # if num_labels == 2:
        #     test_probs = np.array(test_probs)[:, 1]
        # plotter.plot_roc_curve(y_true=test_labels, y_scores=test_probs,
        #                             save_as=f"{task_name.replace(' ', '_')}_roc_curve_{metadata_tag}.png", labels=labels)
        # plotter.plot_precision_recall_curve(y_true=test_labels, y_scores=test_probs,
        #                                             save_as=f"{task_name.replace(' ', '_')}_pr_curve_{metadata_tag}.png", labels=labels)
        # experiment_logger.info(f"Plots saved for {task_name} and {metadata_tag}")

        # experiment_logger.info("ViLT experiment completed.")                        
        # Plot results
    #     plotter = PlottingManager(model_name="VILT", output_dir=args.output_dir)

    #     plotter.plot_confusion_matrix(
    #         y_true=test_labels,
    #         y_pred=test_preds,
    #         labels=labels,
    #         save_as=f"{task_name.replace(' ', '_')}_confusion_matrix.png"
    #     )

    #     # Extract positive class probabilities for binary classification
    #     if num_labels == 2:
    #         test_probs = np.array(test_probs)[:, 1]

    #     plotter.plot_roc_curve(
    #         y_true=test_labels,
    #         y_scores=test_probs,
    #         save_as=f"{task_name.replace(' ', '_')}_roc_curve.png",
    #         labels=labels
    #     )

    #     plotter.plot_precision_recall_curve(
    #         y_true=test_labels,
    #         y_scores=test_probs,
    #         save_as=f"{task_name.replace(' ', '_')}_pr_curve.png",
    #         labels=labels
    #     )

    #     experiment_logger.info(f"Plots saved for {task_name}")

    # experiment_logger.info("ViLT experiment completed.")
def perform_vilt_task(model, train_loader, val_loader, test_loader, test_df,task_name,num_labels, labels, args, folder_manager):
    # Define optimizer and scheduler (you can choose your preferred scheduler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.5, verbose=True)

    # Early stopping parameters
    patience = args.patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Initialize training history dictionary
    training_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_metrics": []
    }

    experiment_logger.info(f"Starting {task_name} training...")
    if getattr(args, "run_lr_finder", False):
        experiment_logger.info("🔍 Running LR Finder for ViLT...")
        best_lr, lrs, losses = lr_finder(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=args.device,
            #save_path=lr_plot_path  # handled by PlottingManager
    )
        # Plot results (using a PlottingManager for VILT).
        plotter = PlottingManager(model_name=args.model.upper(), folder_manager=folder_manager)

        # Save LR Finder plot
        lr_plot_path = os.path.join(folder_manager.images_dir, f"{task_name.replace(' ', '_')}_lr_finder_plot.png")
        plotter.plot_lr_finder(lrs=lrs, losses=losses, save_as=f"{task_name.replace(' ', '_')}_lr_finder_plot.png")
        experiment_logger.info(f"LR Finder plot saved to {lr_plot_path}")
        # Use best LR for training
        args.learning_rate = best_lr
        experiment_logger.info(f"Best learning rate selected from LR Finder: {best_lr:.6f}")

    else:
        experiment_logger.info("Skipping LR Finder (use --run_lr_finder to enable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    for epoch in range(args.epochs):
        experiment_logger.info(f"{task_name} - Epoch {epoch + 1}/{args.epochs}")
        # Train for one epoch using your VILT training function
        epoch_loss, train_losses, train_accuracy = train_vilt(model, train_loader, optimizer, args.device, class_weights=None)
        training_history.setdefault("train_metrics", []).append({"accuracy": train_accuracy})

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(epoch_loss)
        
        # Evaluate on the validation set (using evaluate_vilt)
        val_loss, val_metrics, val_labels, val_preds, val_probs, test_row_indices = evaluate_vilt(model, val_loader, args.device)
        training_history["val_loss"].append(val_loss)
        training_history["val_metrics"].append(val_metrics)
        experiment_logger.info(f"{task_name} - Epoch {epoch+1} complete. Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Checkpoint and early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"
            best_model_path = os.path.join(folder_manager.models_dir, f"{task_name.replace(' ', '_')}_{metadata_tag}_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            experiment_logger.info(f"Best model for {task_name} ({metadata_tag}) saved at {best_model_path}")
        else:
            epochs_without_improvement += 1 
        
        if epochs_without_improvement >= patience:
            experiment_logger.info(f"Early stopping triggered for {task_name}.")
            break

    # Save training history to CSV
    history_df = pd.DataFrame(training_history)
    metadata_tag = "with_metadata" if args.include_metadata else "without_metadata"
    history_filename = f"{task_name.replace(' ', '_')}_training_history_{metadata_tag}.csv"
    history_file = os.path.join(folder_manager.models_dir, history_filename)
    history_df.to_csv(history_file, index=False)
    experiment_logger.info(f"Training history for {task_name} saved to {history_file}")

    # Save the final model
    final_model_filename = f"vilt_final_model_{task_name.replace(' ', '_')}_{metadata_tag}.pth"
    final_model_path = os.path.join(folder_manager.models_dir, final_model_filename)
    torch.save(model.state_dict(), final_model_path)
    experiment_logger.info(f"Final model for {task_name} saved to {final_model_path}")

    
    val_metrics_filename = f"{task_name.replace(' ', '_')}_val_metrics_{metadata_tag}.json"
    plotter.save_metrics(val_metrics, filename=val_metrics_filename)
    experiment_logger.info(f"Validation metrics for {task_name} saved to {val_metrics_filename}")

    plotter.plot_confusion_matrix(y_true=val_labels, y_pred=val_preds, labels=labels,
                              save_as=f"{task_name.replace(' ', '_')}_val_confusion_matrix_{metadata_tag}.png")
    if num_labels == 2:
        val_probs = np.array(val_probs)[:, 1]
    plotter.plot_roc_curve(y_true=val_labels, y_scores=val_probs,
                       save_as=f"{task_name.replace(' ', '_')}_val_roc_curve_{metadata_tag}.png", labels=labels)
    plotter.plot_precision_recall_curve(y_true=val_labels, y_scores=val_probs,
                                    save_as=f"{task_name.replace(' ', '_')}_val_pr_curve_{metadata_tag}.png", labels=labels)


    # Final evaluation on the test set and saving metrics/plots
    experiment_logger.info(f"Evaluating {task_name} on test set...")
    test_loss, test_metrics, test_labels, test_preds, test_probs, test_row_indices = evaluate_vilt(model, test_loader, args.device)
    # Slice test_df to get actual input rows used
    used_test_df = test_df.iloc[test_row_indices].reset_index(drop=True)
    # Build prediction DataFrame
    results_df = pd.DataFrame({
        "true_label": test_labels,
        "predicted_label": test_preds
    })
    # Add class probabilities
    probs_df = pd.DataFrame(np.array(test_probs), columns=[f"prob_class_{i}" for i in range(len(test_probs[0]))])

    # Final join
    final_df = pd.concat([used_test_df.reset_index(drop=True), results_df, probs_df], axis=1)

    # Save under metrics
    pred_save_path = os.path.join(folder_manager.metrics_dir, f"{task_name.replace(' ', '_')}_test_predictions_detailed.csv")
    final_df.to_csv(pred_save_path, index=False)
    experiment_logger.info(f"📝 Saved detailed predictions with inputs to: {pred_save_path}")
    test_metrics_filename = f"{task_name.replace(' ', '_')}_test_metrics_{metadata_tag}.json"
    #save_metrics(test_metrics, folder_manager.metrics_dir, test_metrics_filename)
    experiment_logger.info(f"Test metrics for {task_name} saved to {test_metrics_filename}")
    
    plotter.save_metrics(test_metrics, filename=test_metrics_filename)
    plotter.plot_confusion_matrix(y_true=test_labels, y_pred=test_preds, labels=labels,
                                            save_as=f"{task_name.replace(' ', '_')}_test_confusion_matrix_{metadata_tag}.png")
    if num_labels == 2:
        test_probs = np.array(test_probs)[:, 1]
    plotter.plot_roc_curve(y_true=test_labels, y_scores=test_probs,
                                    save_as=f"{task_name.replace(' ', '_')}_test_roc_curve_{metadata_tag}.png", labels=labels)
    plotter.plot_precision_recall_curve(y_true=test_labels, y_scores=test_probs,
                                                    save_as=f"{task_name.replace(' ', '_')}_test_pr_curve_{metadata_tag}.png", labels=labels)
    experiment_logger.info(f"Plots saved for {task_name} and {metadata_tag}")

    # Plot the loss curve after training ends
    plotter.plot_training_curves(
        train_losses=training_history["train_loss"], 
        val_losses=training_history["val_loss"], 
        train_accuracies=[metric["accuracy"] for metric in training_history["train_metrics"]],
        val_accuracies=[metric["accuracy"] for metric in training_history["val_metrics"]],
        save_as=f"{task_name.replace(' ', '_')}_training_validation_curve_{metadata_tag}.png"
    )
    experiment_logger.info(f"Plots saved for {task_name} and {metadata_tag}")

    experiment_logger.info("ViLT experiment completed.")  


def run_clip_experiment(train_df, val_df, test_df, args):
    """
    Main function to train and evaluate CLIP for 2-way and 3-way classification.

    Args:
        train_df (DataFrame): Training dataset.
        val_df (DataFrame): Validation dataset.
        test_df (DataFrame): Test dataset.
        args (Namespace): Experiment arguments/configurations.
    """
    try:
        # Initialize CLIP processor and dataloaders
        experiment_logger.info("Initializing CLIP processor and data loaders...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_loader = get_clip_dataloader(train_df, processor, args.metadata_columns, args.batch_size, shuffle=True)
        val_loader = get_clip_dataloader(val_df, processor, args.metadata_columns, args.batch_size)
        test_loader = get_clip_dataloader(test_df, processor, args.metadata_columns, args.batch_size)

        # Log dataset details
        experiment_logger.info(f"Dataset sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        experiment_logger.info(f"Metadata columns used: {args.metadata_columns}")

        # Initialize CLIP model
        experiment_logger.info("Initializing CLIP model...")
        model = CLIPMultiTaskClassifier(
            input_dim=args.input_dim,
            num_classes_2=args.num_classes_2,
            num_classes_3=args.num_classes_3,
            metadata_dim=len(args.metadata_columns)
        ).to(args.device)

        # Log model details
        experiment_logger.info(f"Model initialized with input_dim={args.input_dim}, "
                                f"num_classes_2={args.num_classes_2}, num_classes_3={args.num_classes_3}, "
                                f"metadata_dim={len(args.metadata_columns)}.")

        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        experiment_logger.info(f"Optimizer initialized with learning rate: {args.learning_rate}")

        # Training phase
        experiment_logger.info("Starting training...")
        train_loss = train_model(model, train_loader, optimizer, args.device)
        experiment_logger.info(f"Training completed. Final training loss: {train_loss:.4f}")

        # Save model checkpoint
        model_checkpoint_path = os.path.join(args.output_dir, "clip_model_checkpoint.pth")
        torch.save(model.state_dict(), model_checkpoint_path)
        experiment_logger.info(f"Model checkpoint saved at {model_checkpoint_path}")

        # Evaluation phase
        experiment_logger.info("Starting evaluation...")
        evaluate_and_save_metrics(model, val_loader, test_loader, args)

        experiment_logger.info("CLIP experiment completed successfully.")

    except Exception as e:
        experiment_logger.error(f"Error during CLIP experiment: {str(e)}")
        raise

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
            #logger.warning("Image has very low variance; likely a placeholder.")
            return True

        # Optional: OCR-based detection
        try:
            text = pytesseract.image_to_string(img)
            placeholder_keywords = [
                "not found", "unavailable", "error", "placeholder", "no longer available"
            ]
            for keyword in placeholder_keywords:
                if keyword.lower() in text.lower():
                    #logger.warning(f"Detected placeholder keyword '{keyword}' in image.")
                    return True
        except Exception as e:
            experiment_logger.warning(f"OCR processing failed: {e}")

    except Exception as e:
        #logger.error(f"Error during placeholder detection: {e}")
        # If something goes wrong, better treat it as a placeholder to skip it.
        return True

    return False

def preprocess_image(url_or_path, size=(224, 224)):
    """
    Load and preprocess an image from either a local path or a URL.
    Returns a resized RGB PIL image, or None if the image cannot be loaded.
    """
    import warnings, os
    from urllib.parse import urlparse
    from PIL import Image
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO

    try:
        # ✅ 1. Check if it's a local file
        if os.path.exists(url_or_path):
            img = Image.open(url_or_path).convert("RGB")

        # ✅ 2. Otherwise, assume it's a URL and fetch it
        else:
            parsed = urlparse(url_or_path)
            if not parsed.scheme:
                raise ValueError(f"Invalid URL or path: {url_or_path}")

            session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
            session.mount('http://', HTTPAdapter(max_retries=retries))
            session.mount('https://', HTTPAdapter(max_retries=retries))
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                              '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'Referer': url_or_path,
            })
            response = session.get(url_or_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")

        # ✅ 3. Handle empty or placeholder images
        if img.size == (0, 0):
            warnings.warn(f"Empty image: {url_or_path}")
            return None

        if is_placeholder_image(img):
            warnings.warn(f"Placeholder image detected at: {url_or_path}")
            return None

        # ✅ 4. Resize and return
        return img.resize(size)

    except requests.exceptions.HTTPError as http_err:
        warnings.warn(f"HTTP error for image {url_or_path}: {http_err}")
    except requests.RequestException as req_err:
        warnings.warn(f"Request error for image {url_or_path}: {req_err}")
    except Exception as e:
        warnings.warn(f"Error processing image {url_or_path}: {e}")
    return None
