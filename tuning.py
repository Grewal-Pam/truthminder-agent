# Initial Setup and Imports
import os
import logging
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor, ViltModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
import random
import pickle
# Perform hyperparameter tuning
import optuna

def objective(trial, data_loaders, class_weights, label_type, num_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    epochs = trial.suggest_int('epochs', 1, 5)
    
    model = ViltClassificationModel(vilt_model, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    
    for epoch in range(epochs):
        train_loss, train_losses, train_accuracies = train_vilt(model, data_loaders[label_type]['train'], optimizer, device, class_weights[label_type])
        val_loss, val_accuracy, val_kappa, val_f1, val_recall, val_precision, _, _, val_losses, val_accuracies = evaluate_vilt(model, data_loaders[label_type]['val'], device)
        
        all_train_losses.append(np.mean(train_losses))
        all_val_losses.append(np.mean(val_losses))
        all_train_accuracies.append(np.mean(train_accuracies))
        all_val_accuracies.append(np.mean(val_accuracies))
    
    return val_accuracy  # Return the validation accuracy for the trial

def tune_hyperparameters(label_columns, data_loaders, class_weights):
    best_params = {}
    for label_type in label_columns:
        num_labels = train_df[label_type].nunique()  # Determine the number of unique labels
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, data_loaders, class_weights, label_type, num_labels), n_trials=10)
        
        best_params[label_type] = study.best_params
        logging.info(f"Best params for {label_type}: {study.best_params}")
        
        # Save the study results
        study_path = os.path.join(FEATURE_DIR, f"optuna_study_{label_type}.pkl")
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        logging.info(f"Optuna study for {label_type} saved to {study_path}")
    
    return best_params


# Toggles
USE_HYPERPARAMETER_TUNING = False
INCLUDE_METADATA = True

if USE_HYPERPARAMETER_TUNING:
    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(['2_way_label', '3_way_label'], data_loaders, class_weights) #, '6_way_label'
else:
    best_params = None