# 🧠 TruthMinder-Agent  
**A Multimodal Disinformation Detection System using CLIP, ViLT, and FLAVA**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### 🚀 Overview
**TruthMinder-Agent** is an agentic AI system designed to detect **misinformation** in online news posts using a **multimodal pipeline** that analyzes both **text and images**, supported by **metadata** and **logical consistency checks**.

This project extends Parminder Grewal’s **Master’s Thesis in Web and Data Science (University of Koblenz, 2025)** — accepted for publication at **ACM DHOW 2025 (Dublin)** — and integrates the best-performing models (CLIP, ViLT, FLAVA) into an interactive **Streamlit dashboard**.

---

### 🧩 Core Features
- **Multimodal Fusion** of text, image, and post metadata  
- **Ensemble of Pretrained Models** – CLIP, ViLT, and FLAVA  
- **Evidence Layer** – OCR-based text extraction from images  
- **Consistency Layer** – Natural Language Inference (NLI) for logical checks  
- **Arbiter Module** – Final decision fusion with abstain/uncertain handling  
- **Interactive UI** – Streamlit-based visualization and test interface  

---
## 🧭 System Architecture

The TruthMinder-Agent follows a modular **agentic workflow** consisting of four main reasoning stages:

```mermaid
flowchart LR
    A["🧠 Perception<br/>CLIP · ViLT · FLAVA"] --> B["📜 Evidence<br/>OCR Extraction"]
    B --> C["🔍 Consistency<br/>NLI + CLIP Similarity"]
    C --> D["⚖️ Arbiter<br/>Decision Fusion"]
    D --> E["✅ Final Label<br/>(Real · Satire/Mixed · Fake)"]

## 🧩 Core Components

The TruthMinder-Agent project is structured into clear modular layers to ensure transparency, maintainability, and reproducibility.

| Folder / File | Description |
|----------------|-------------|
| **agent/** | Core orchestration logic for the multimodal AI agent. Contains `runner.py`, which manages the Perception → Evidence → Consistency → Arbiter workflow. |
| **models/** | Inference scripts for the three pretrained transformer models — **CLIP**, **ViLT**, and **FLAVA** — each adapted for binary (2-way) and ternary (3-way) disinformation classification. |
| **tools/** | Utility modules for auxiliary reasoning: **OCR extraction**, **NLI consistency checking**, **cosine similarity**, and **explainability** utilities. |
| **app/** | Streamlit-based user interface where users can upload, analyze, and visualize results interactively. |
| **evaluate/** | Scripts to regenerate and evaluate model metrics (accuracy, F1-score, precision, recall, Cohen’s Kappa) from checkpoints. |
| **results/** | Stores evaluation plots such as confusion matrices, ROC and Precision-Recall curves for CLIP, ViLT, and FLAVA models. |
| **outputs/** | Contains final agent outputs — including `enriched.csv` (batch predictions) and JSON traces for each processed post. |
| **training/** | Training utilities including learning rate finder, trainers, and evaluators used during model fine-tuning. |
| **datasets/** | Preprocessing and dataset construction scripts for the Fakeddit dataset. Handles text cleaning, label encoding, and metadata normalization. |
| **runs/** | TensorBoard-compatible experiment logs for model training sessions. |
| **requirements.txt** | Python dependencies required to run the project. |
| **README.md** | This documentation file describing the project’s structure and purpose. |

---

### 🧠 Highlights
- Modularized architecture enables individual testing of CLIP, ViLT, and FLAVA.
- Each layer (Perception → Evidence → Consistency → Arbiter) is fully traceable through JSON logs.
- Enables both **single-post analysis** (interactive UI) and **batch evaluation** (CLI mode).

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/parmindergrewal/truthminder-agent.git
cd truthminder-agent

2️⃣ Create and Activate a Conda Environment

It’s recommended to use a fresh Python 3.9 environment for compatibility with PyTorch and Hugging Face Transformers.

conda create -n truthminder python=3.9 -y
conda activate truthminder

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Verify Installation
python -c "from tools import ocr; print('✅ OCR module import OK')"


You should see:

✅ OCR module import OK


🧠 Running the Agentic Pipeline
▶️ Option 1: Run via Streamlit (Interactive UI)

Launch the complete web app locally:

streamlit run app/test_app.py --server.port 8501


Once launched, open your browser and go to:
👉 http://localhost:8501

This allows you to:

Upload a post (image + caption)

View predictions from CLIP, ViLT, and FLAVA

Inspect OCR, NLI, and Arbiter reasoning

See the final disinformation label with confidence

▶️ Option 2: Run in Background (Server Mode)

Useful for remote VMs or cloud environments:

nohup streamlit run app/test_app.py --server.port 8501 > truthmindr.log 2>&1 &


Check logs:

tail -f truthmindr.log


Stop the process:

pkill -f streamlit

▶️ Option 3: Batch Mode (Offline Evaluation)

To process a file containing multiple posts:

python agent/runner.py


This will create:

outputs/enriched.csv — enriched predictions

outputs/traces/*.json — per-post reasoning trace files