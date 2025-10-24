import torch

import optuna
from training.flava_training import train_flava
from evaluate.flava_evaluate import evaluate_flava
from utils.logger import setup_logger

logger = setup_logger("flava_tuning_log")


# Objective function
def objective(trial, model, train_loader, val_loader, device, class_weights):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model for the trial
    train_flava(model, train_loader, optimizer, device, class_weights)

    # Evaluate on the validation set
    metrics, _ = evaluate_flava(model, val_loader, device)

    # Return the metric to maximize
    return metrics["accuracy"]


# Hyperparameter tuning function
def tune_hyperparameters(
    model,
    train_loader,
    val_loader,
    device,
    class_weights,
    task_name="2-way",
    n_trials=10,
):
    logger.info(f"Starting hyperparameter tuning for {task_name} classification...")
    study_name = f"{task_name}_classification_tuning"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(
        lambda trial: objective(
            trial, model, train_loader, val_loader, device, class_weights
        ),
        n_trials=n_trials,
    )
    return study.best_params


# # Validate labels
# def validate_labels(dataloader, num_labels):
#     for batch in dataloader:
#         if batch is None:  # Skip empty batches
#             continue
#         labels = batch['labels']
#         if torch.any(labels >= num_labels) or torch.any(labels < 0):
#             raise ValueError(f"Found out-of-bounds labels in the dataset: {labels}")

# # Validate labels for the 6-way classification
# #validate_labels(data_loaders['6_way_label']['train'], 6)
# #validate_labels(data_loaders['6_way_label']['val'], 6)
# #validate_labels(data_loaders['6_way_label']['test'], 6)

# # Perform hyperparameter tuning
# def objective(trial, data_loaders, class_weights, label_type, num_labels):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Define hyperparameters to tune
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
#     epochs = trial.suggest_int('epochs', 1, 5)

#     model = FlavaClassificationModel(flava_model, num_labels, include_metadata=INCLUDE_METADATA).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#     all_train_losses = []
#     all_val_losses = []
#     all_train_accuracies = []
#     all_val_accuracies = []

#     for epoch in range(epochs):
#         train_loss, train_losses, train_accuracies = train_flava(model, data_loaders[label_type]['train'], optimizer, device, class_weights[label_type])
#         val_loss, val_accuracy, val_kappa, val_f1, val_recall, val_precision, _, _, val_losses, val_accuracies = evaluate_flava(model, data_loaders[label_type]['val'], device)

#         all_train_losses.append(np.mean(train_losses))
#         all_val_losses.append(np.mean(val_losses))
#         all_train_accuracies.append(np.mean(train_accuracies))
#         all_val_accuracies.append(np.mean(val_accuracies))

#     return val_accuracy  # Return the validation accuracy for the trial

# def tune_hyperparameters(label_columns, data_loaders, class_weights):
#     best_params = {}
#     for label_type in label_columns:
#         num_labels = train_df[label_type].nunique()  # Determine the number of unique labels
#         study = optuna.create_study(direction='maximize')
#         study.optimize(lambda trial: objective(trial, data_loaders, class_weights, label_type, num_labels), n_trials=10)

#         best_params[label_type] = study.best_params
#         logging.info(f"Best params for {label_type}: {study.best_params}")

#         # Save the study results
#         study_path = os.path.join(FEATURE_DIR, f"optuna_study_{label_type}.pkl")
#         with open(study_path, 'wb') as f:
#             pickle.dump(study, f)
#         logging.info(f"Optuna study for {label_type} saved to {study_path}")

#     return best_params

# if USE_HYPERPARAMETER_TUNING:
#     print("Tuning hyperparameters...")
#     best_params = tune_hyperparameters(['2_way_label', '3_way_label'], data_loaders, class_weights) #, '6_way_label'
# else:
#     best_params = None

# # Train and evaluate model
# #epochs = 3
