import os
import json
import mlflow
from datetime import datetime

# Define base runs folder
RUNS_DIR = "runs"
EXPERIMENT_NAME = "truthmindr_experiments"

# Only keep latest N experiments per model
KEEP_LAST_N = 3

# Allowed artifact extensions (safe, small)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".json"}

def list_model_runs(model_dir):
    """
    List timestamped subfolders (each representing a run),
    sorted by modified time descending.
    """
    runs = []
    for run_name in os.listdir(model_dir):
        path = os.path.join(model_dir, run_name)
        if os.path.isdir(path):
            runs.append((run_name, os.path.getmtime(path)))
    # Sort by time (newest first)
    runs.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in runs[:KEEP_LAST_N]]  # keep only latest N


def log_run_to_mlflow(model_name, run_path):
    meta_path = os.path.join(run_path, "run_metadata.json")
    metrics_path = os.path.join(run_path, "metrics")

    if not os.path.exists(meta_path):
        print(f"⚠️ Skipping {run_path}: no run_metadata.json found.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Start MLflow run
    model_name = model_name.upper()
    with mlflow.start_run(run_name=f"{model_name}_{meta.get('timestamp', datetime.now().isoformat())}"):
        # Log basic parameters
        mlflow.log_param("model", model_name)
        for k, v in meta.items():
            # prevent duplicate param keys (case-insensitive)
            if k.lower() == "model":
                continue
            mlflow.log_param(k, v)

        # Log metrics from JSON files
        if os.path.exists(metrics_path):
            for file in os.listdir(metrics_path):
                if file.endswith(".json"):
                    with open(os.path.join(metrics_path, file), "r") as f:
                        data = json.load(f)
                    for k, v in data.items():
                        try:
                            mlflow.log_metric(k, float(v))
                        except Exception:
                            pass

        # Log small artifacts (images, JSONs only)
        for root, _, files in os.walk(run_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    try:
                        mlflow.log_artifact(full_path)
                    except Exception as e:
                        print(f"⚠️ Skipped {file}: {e}")


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name in ["CLIP", "VILT", "FLAVA"]:
        model_dir = os.path.join(RUNS_DIR, model_name)
        if not os.path.exists(model_dir):
            print(f"⚠️ No folder for {model_name}, skipping.")
            continue

        print(f"📂 Processing model: {model_name}")
        for run_name in list_model_runs(model_dir):
            run_path = os.path.join(model_dir, run_name)
            print(f"  🔹 Logging {run_name} ...")
            log_run_to_mlflow(model_name, run_path)

    print("\n✅ Backfill complete! View results via:  mlflow ui --port 5000")
