from __future__ import annotations

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.model_loader import load_models, run_model_inference
from app.schemas import PredictionItem, PredictionResponse

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
CLASS_NAMES_PATH = Path("/models/class_names.json")
models: dict[str, Any] = {}
class_names: list[str] = []

GTSRB_CLASS_LABELS: dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons",
}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    return np.expand_dims(image_array, axis=0)

def extract_top_prediction(predictions: np.ndarray) -> tuple[int, str]:
    output = np.asarray(predictions)

    if output.ndim == 0:
        score = float(output)
        return 0, f"{score:.6f}"

    if output.ndim > 1:
        output = output[0]

    class_idx = int(np.argmax(output))
    confidence = float(output[class_idx])
    return class_idx, f"{confidence:.6f}"


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        LOGGER.warning("Class name mapping not found at %s", CLASS_NAMES_PATH)
        return []

    try:
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Failed reading class name mapping: %s", exc)
        return []

    if not isinstance(raw, list):
        LOGGER.warning("Invalid class name mapping format at %s", CLASS_NAMES_PATH)
        return []

    parsed = [str(item) for item in raw]
    LOGGER.info("Loaded %d class names from %s", len(parsed), CLASS_NAMES_PATH)
    return parsed


def resolve_dataset_class_id(output_class_idx: int, class_name_list: list[str]) -> str:
    if 0 <= output_class_idx < len(class_name_list):
        candidate = class_name_list[output_class_idx].strip()
        if candidate:
            return candidate
    return str(output_class_idx)


def resolve_class_name(class_id: str) -> str:
    try:
        idx = int(class_id)
    except ValueError:
        return class_id

    if idx in GTSRB_CLASS_LABELS:
        return GTSRB_CLASS_LABELS[idx]
    if 1 <= idx <= 43 and (idx - 1) in GTSRB_CLASS_LABELS:
        return GTSRB_CLASS_LABELS[idx - 1]
    return f"Class {class_id}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, class_names
    LOGGER.info("Loading TensorFlow models at startup")
    models = load_models()
    class_names = load_class_names()
    app.state.models = models
    app.state.class_names = class_names
    LOGGER.info("Loaded %d models", len(models))
    yield


app = FastAPI(
    title="TensorFlow Model Comparison Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow local frontend dev servers to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    image_batch = preprocess_image(image_bytes)

    results: list[PredictionItem] = []
    for model_name, model in request.app.state.models.items():
        start_time = time.perf_counter()
        # Models already contain a preprocessing layer from training; do not preprocess again.
        raw_predictions = run_model_inference(model, image_batch)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        output_class_idx, confidence = extract_top_prediction(raw_predictions)
        class_id = resolve_dataset_class_id(output_class_idx, request.app.state.class_names)
        results.append(
            PredictionItem(
                model=model_name,
                class_id=class_id,
                class_name=resolve_class_name(class_id),
                confidence=confidence,
                inference_time_ms=f"{elapsed_ms:.3f}",
            )
        )

    return PredictionResponse(filename=file.filename or "", predictions=results)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
