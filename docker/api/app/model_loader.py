from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger(__name__)

MODELS_ROOT = Path("/models")
MODEL_NAMES = ("VGG16", "ResNet50", "MobileNetV3Small")


def _load_single_model(model_dir: Path) -> Any:
    # Prefer Keras load_model for SavedModel exports generated from Keras.
    try:
        return tf.keras.models.load_model(model_dir)
    except Exception as keras_exc:
        LOGGER.warning("Keras loader failed for %s: %s; trying tf.saved_model.load", model_dir, keras_exc)
        return tf.saved_model.load(model_dir)


def load_models() -> dict[str, Any]:
    loaded_models: dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        model_path = MODELS_ROOT / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        loaded_models[model_name] = _load_single_model(model_path)
        LOGGER.info("Loaded model %s from %s", model_name, model_path)

    return loaded_models


def run_model_inference(model: Any, input_batch: np.ndarray) -> np.ndarray:
    tensor_input = tf.convert_to_tensor(input_batch, dtype=tf.float32)

    if hasattr(model, "predict"):
        outputs = model.predict(input_batch, verbose=0)
    elif hasattr(model, "signatures") and "serving_default" in model.signatures:
        signature_output = model.signatures["serving_default"](tensor_input)
        outputs = next(iter(signature_output.values()))
    else:
        try:
            outputs = model(tensor_input, training=False)
        except TypeError:
            outputs = model(tensor_input)

    if isinstance(outputs, dict):
        outputs = next(iter(outputs.values()))
    elif isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    if isinstance(outputs, tf.Tensor):
        outputs = outputs.numpy()

    return np.asarray(outputs)
