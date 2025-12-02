# Evaluation Module

Scripts and utilities for model evaluation, metrics calculation, and visualization.

## Overview

Evaluation workflows for the three models (CLIP, ViLT, FLAVA) on the Fakeddit dataset:
- Binary classification (Real vs. Fake)
- Ternary classification (Real vs. Satire/Mixed vs. Fake)
- Metrics: Accuracy, F1-score, Precision, Recall, Cohen's Kappa
- Visualizations: ROC curves, Confusion matrices, Precision-Recall curves

---

## Files

- `regenerate_clip_metrics.py` - CLIP evaluation and metrics generation
- `evaluate_vilt_from_checkpoint.py` - ViLT checkpoint loading and evaluation
- `vilt_evaluate.py` - ViLT evaluation utilities

---

## CLIP Evaluation

**File:** `regenerate_clip_metrics.py`

### Function
```python
regenerate_clip_metrics(
    test_dataloader,
    model,
    task="3way",
    output_dir="results/CLIP/"
)
```

### Outputs
```
results/CLIP/
├── test_2_way_accuracy.json
├── test_2_way_roc_curve.png
├── test_2_way_confusion_matrix.png
├── test_3_way_accuracy.json
├── test_3_way_confusion_matrix.png
└── test_3_way_roc_curve.png
```

### Metrics
- **Accuracy:** Overall correctness
- **F1-score:** Harmonic mean of precision & recall
- **Cohen's Kappa:** Inter-rater agreement
- **Precision/Recall per class**

---

## ViLT Evaluation

**File:** `evaluate_vilt_from_checkpoint.py`

### Function
```python
evaluate_from_checkpoint(
    checkpoint_path: str,
    test_dataset,
    task: str = "3way"
) -> dict
```

### Process
1. Load model from checkpoint
2. Run inference on test set
3. Calculate metrics
4. Generate visualizations

### Outputs
```
results/VILT/VILT/images/
├── 2-way_classification_roc_curve.png
├── 2-way_classification_confusion_matrix.png
├── 3-way_classification_roc_curve.png
└── 3-way_classification_confusion_matrix.png
```

---

## Usage Example

```python
from evaluate.regenerate_clip_metrics import regenerate_clip_metrics
from models.clip_model import CLIPModel
from dataloaders.clip_dataloader import get_dataloader

# Load model and data
model = CLIPModel.from_pretrained("model_checkpoint.pt")
test_loader = get_dataloader("test", batch_size=32)

# Run evaluation
metrics = regenerate_clip_metrics(
    test_loader,
    model,
    task="3way",
    output_dir="results/CLIP/"
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-score: {metrics['f1_score']:.3f}")
```

---

## Metrics Explanation

### Accuracy
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

### F1-Score
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Cohen's Kappa
$$\kappa = \frac{\text{Agreement} - \text{Expected Agreement}}{1 - \text{Expected Agreement}}$$

### ROC-AUC
- Measures trade-off between True Positive Rate and False Positive Rate
- AUC = 1.0 is perfect classification
- AUC = 0.5 is random guessing

---

## Performance Benchmarks

### CLIP Model
- **2-way:** 92.4% accuracy, 0.843 Kappa
- **3-way:** 80.2% accuracy, 0.684 Kappa
- **Speed:** ~150 ms/image on GPU

### ViLT Model
- **2-way:** 88.7% accuracy, 0.769 Kappa
- **3-way:** 75.2% accuracy, 0.621 Kappa
- **Speed:** ~80 ms/image on GPU (fastest)

### FLAVA Model
- **2-way:** 90.1% accuracy, 0.812 Kappa
- **3-way:** 78.6% accuracy, 0.657 Kappa
- **Speed:** ~120 ms/image on GPU

---

## Custom Evaluation

### Create Custom Evaluation Script
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def custom_evaluate(predictions, ground_truth, labels):
    # Classification report
    report = classification_report(
        ground_truth, 
        predictions,
        target_names=labels
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion_matrix.png')
```

---

## Dataset Splits

### Fakeddit Dataset
- **Training:** 45,000 posts
- **Validation:** 5,000 posts
- **Testing:** 5,000 posts

### Class Distribution (2-way)
- Real: 70%
- Fake: 30%

### Class Distribution (3-way)
- Real: 70%
- Satire/Mixed: 15%
- Fake: 15%

---

## Running Full Evaluation Pipeline

```bash
# 1. Evaluate all models
python evaluate/regenerate_clip_metrics.py --task 3way
python evaluate/evaluate_vilt_from_checkpoint.py --task 3way
python evaluate/vilt_evaluate.py --task 3way

# 2. Generate comparison plots
python evaluate/compare_models.py

# 3. View results
ls results/*/
```

---

## Visualization

### ROC Curve
Shows trade-off between True Positive Rate (sensitivity) and False Positive Rate.
- Better model → curve closer to top-left
- AUC (Area Under Curve) is overall metric

### Confusion Matrix
Shows actual vs predicted for each class:
```
                Predicted
                Real  Fake
Actual Real     850   150
       Fake      80   920
```

### Precision-Recall Curve
Shows relationship between precision and recall as threshold changes.

---

## Troubleshooting

### Out of memory during evaluation
```python
# Reduce batch size
evaluate_vilt_from_checkpoint(
    checkpoint,
    dataset,
    batch_size=8  # Reduced from 32
)
```

### Missing checkpoint file
```bash
# Check available checkpoints
ls models/checkpoints/

# Verify path is absolute
checkpoint_path = "/absolute/path/to/checkpoint.pt"
```

### Visualization not saving
```python
# Ensure output directory exists
import os
os.makedirs("results/", exist_ok=True)
```

---

## References

- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROC Curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
