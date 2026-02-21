from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from app.model_loader import load_models, run_model_inference
from app.schemas import PredictionItem, PredictionResponse

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
models: dict[str, Any] = {}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    return np.expand_dims(image_array, axis=0)


def preprocess_for_model(model_name: str, image_batch: np.ndarray) -> np.ndarray:
    model_input = image_batch.copy()

    if model_name == "VGG16":
        return vgg16_preprocess(model_input)
    if model_name == "ResNet50":
        return resnet50_preprocess(model_input)
    if model_name == "MobileNetV3Small":
        return mobilenet_preprocess(model_input)

    return model_input


def extract_top_prediction(predictions: np.ndarray) -> tuple[str, str]:
    output = np.asarray(predictions)

    if output.ndim == 0:
        score = float(output)
        return "0", f"{score:.6f}"

    if output.ndim > 1:
        output = output[0]

    class_idx = int(np.argmax(output))
    confidence = float(output[class_idx])
    return str(class_idx), f"{confidence:.6f}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    LOGGER.info("Loading TensorFlow models at startup")
    models = load_models()
    app.state.models = models
    LOGGER.info("Loaded %d models", len(models))
    yield


app = FastAPI(
    title="TensorFlow Model Comparison Inference API",
    version="1.0.0",
    lifespan=lifespan,
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
        model_input = preprocess_for_model(model_name, image_batch)
        start_time = time.perf_counter()
        raw_predictions = run_model_inference(model, model_input)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        class_id, confidence = extract_top_prediction(raw_predictions)
        results.append(
            PredictionItem(
                model=model_name,
                class_id=class_id,
                confidence=confidence,
                inference_time_ms=f"{elapsed_ms:.3f}",
            )
        )

    return PredictionResponse(filename=file.filename or "", predictions=results)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
