import argparse
import logging
import torch
from experiment_manager import run_experiment
from utils.folder_manager import ExperimentFolderManager


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run multimodal experiments")
    # parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (TSV format)")
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/TRAIN DATA/train_data.tsv",
        help="Path to the training dataset file (TSV format)",
    )
    parser.add_argument(
        "--validate_path",
        type=str,
        default="data/VALIDATE DATA/validate_data.tsv",
        help="Path to the validation dataset file (TSV format)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/TEST DATA/test_data.tsv",
        help="Path to the test dataset file (TSV format)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs", help="Base directory to save results"
    )
    # parser.add_argument("--log_dir", type=str, default="logs", help="Base directory to save logs")
    parser.add_argument(
        "--input_dim", type=int, default=512, help="Input dimension for the model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["clip", "vilt", "flava"],
        help="Model to use (e.g., clip, vilt, flava)",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Include metadata in the training process",
    )  # Added this argument
    parser.add_argument(
        "--use_tuning", action="store_true", help="Use hyperparameter tuning"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="If set, retrains the model even if a saved model exists.",
    )
    parser.add_argument(
        "--base_dir", type=str, default="runs", help="Base directory for saving results"
    )
    parser.add_argument(
        "--fine_tune", action="store_true", help="Fine-tune the pre-trained weights"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before early stopping",
    )
    parser.add_argument(
        "--run_lr_finder", action="store_true", help="Run LR Finder before training"
    )

    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Stop after preprocessing datasets",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a small subset of the data for debugging (10 train, 5 val, 5 test).",
    )

    args = parser.parse_args()

    # Additional configurations
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.metadata_columns = ["num_comments", "score", "upvote_ratio"]
    args.num_classes_2 = 2
    args.num_classes_3 = 3
    args.labels_2 = ["0", "1"]
    args.labels_3 = ["0", "1", "2"]

    # base_dir = "runs"  # New base directory for experiments
    folder_manager = ExperimentFolderManager(
        base_dir=args.output_dir, model_name=args.model
    )
    args.log_dir = folder_manager.logs_dir
    # Save metadata
    metadata = {
        "timestamp": folder_manager.timestamp,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device.type,
        "include_metadata": args.include_metadata,
        "fine_tune": args.fine_tune,
    }
    folder_manager.save_metadata(metadata)

    # Set up logging
    log_file = folder_manager.get_log_file(args.model)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Experiment started.")

    # Organize directories by model
    # args.log_dir = os.path.join(args.log_dir, args.model.upper())  # e.g., logs/CLIP/
    # args.output_dir = os.path.join(args.output_dir, args.model.upper())  # e.g., results/CLIP/
    # os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the PlottingManager
    # plotter = PlottingManager(model_name=args.model.upper(), output_dir=args.output_dir)

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
