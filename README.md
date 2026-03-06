# CNN Model Comparison for Traffic Sign Recognition

This repository contains the implementation for a Finnish *Ammattikorkeakoulu* (University of Applied Sciences) thesis project.

Thesis objective: identify the best-performing CNN architecture for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The project compares three CNN-based transfer learning models:
- `VGG16`
- `ResNet50`
- `MobileNetV3Small`

The repository includes:
- model training pipeline (TensorFlow, Docker)
- inference API (FastAPI)
- web frontend for model comparison (React + TypeScript)
- report generation scripts for thesis figures and tables

## Thesis Scope

The thesis evaluates which model is the best overall choice on GTSRB by comparing:
- classification accuracy and loss
- training time
- parameter count
- inference latency
- class-level error patterns (confusion matrices and top confusions)

Dataset: German Traffic Sign Recognition Benchmark (GTSRB), 43 traffic sign classes.

## Repository Structure

```text
.
├── docker/
│   ├── training/          # TensorFlow training image + train.py
│   └── api/               # FastAPI inference service
├── frontend/              # Thesis demo UI (React + Vite + Tailwind)
├── reports/               # Report generation script + docs
├── data/                  # Expected dataset mount point (Train/Test)
├── outputs/               # Generated models and metrics (ignored by git)
└── create_dummy_models.py # Utility for generating placeholder models
```

## End-to-End Workflow

1. Prepare dataset in `data/Train` and `data/Test`.
2. Train models with Docker (`docker/training/train.py`).
3. Training outputs are written under `outputs/models/`.
4. Generate thesis artifacts with `reports/make_report.py`.
5. Serve trained models through FastAPI.
6. Use frontend to upload an image and compare model predictions.

## Prerequisites

- Docker (recommended for training + API)
- Node.js 18+ (for frontend)
- Python 3.10+ (only needed for local report generation)
- Optional but recommended for training: NVIDIA GPU + NVIDIA Container Toolkit

## Dataset Layout

Expected directory structure:

```text
data/
├── Train/
│   ├── 0/
│   ├── 1/
│   └── ...
└── Test/
    ├── 0/
    ├── 1/
    └── ...
```

Notes:
- `Train` and `Test` must contain the same class directories.
- Current training script expects `43` classes (GTSRB).

## Training (Docker)

Build image:

```bash
docker build -t thesis-training -f docker/training/Dockerfile docker/training
```

Run training (GPU):

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  thesis-training
```

If you do not have GPU support configured, try without `--gpus all` (training will be slower).

Training writes:
- `outputs/models/<MODEL_NAME>/` (SavedModel, history, metrics, predictions)
- `outputs/models/class_names.json`
- `outputs/models/comparison.csv`

## Generate Thesis Reports

Install minimal report dependencies:

```bash
python -m pip install -r docker/training/requirements.txt
```

Generate report artifacts:

```bash
python reports/make_report.py
```

Outputs are written to `reports/`:
- `summary.csv`
- `figures/` (accuracy, inference time, parameter count, learning curves, confusion matrices)
- `tables/` (confusion matrix CSVs)
- `error_analysis.csv`
- `top_confusions.csv`

More details: [`reports/README.md`](reports/README.md)

## Inference API (Docker)

Build API image:

```bash
docker build -t thesis-api -f docker/api/Dockerfile docker/api
```

Run API with trained models mounted read-only:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/outputs/models:/models:ro" \
  thesis-api
```

Available endpoints:
- `GET /healthz`
- `POST /predict` with `multipart/form-data` (`file` field)

API expects:
- `/models/VGG16`
- `/models/ResNet50`
- `/models/MobileNetV3Small`
- `/models/class_names.json` (optional but recommended)

## Frontend (Local)

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```

Open the printed Vite URL (usually `http://localhost:5173`).

The UI supports:
- image upload / drag-and-drop
- optional crop before inference
- side-by-side model prediction comparison
- confidence and latency charts

More details: [`frontend/README.md`](frontend/README.md)

## Reproducibility Notes

- Training uses fixed seeds (`SEED = 42`) and deterministic class ordering.
- Class mapping used in training is exported to `outputs/models/class_names.json`.
- Generated artifacts in `outputs/` and `reports/figures`, `reports/tables` are ignored by git.

## License

No license file is currently defined in this repository.
