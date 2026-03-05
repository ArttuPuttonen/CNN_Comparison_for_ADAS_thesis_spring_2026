#!/usr/bin/env python3
"""Train and compare multiple transfer-learning models on an image dataset.

Expected dataset layout:
  /data/Train/<class_name>/*
  /data/Test/<class_name>/*

Outputs:
  /outputs/models/<MODEL_NAME>/
    - TensorFlow SavedModel
    - history.json
    - metrics.txt
    - predictions.csv
  /outputs/models/class_names.json
  /outputs/models/comparison.csv
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, mixed_precision, models, optimizers
from tensorflow.keras.applications import (
    MobileNetV3Small,
    ResNet50,
    VGG16,
)
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path("/data")
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
OUTPUT_ROOT = Path("/outputs/models")

IMG_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
BATCH_SIZE: int = 64
NUM_CLASSES: int = 43

INITIAL_EPOCHS: int = 15
FINE_TUNE_EPOCHS: int = 15
INITIAL_LR: float = 1e-3
FINE_TUNE_LR: float = 1e-5
UNFREEZE_TOP_LAYERS: int = 30
SEED: int = 42

AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DS: Optional[tf.data.Dataset] = None
TEST_DS: Optional[tf.data.Dataset] = None
TEST_IMAGE_COUNT: Optional[int] = None
CLASS_NAMES: List[str] = []


def set_reproducibility(seed: int = SEED) -> None:
    """Set deterministic seeds where possible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def prepare_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load and optimize datasets using image_dataset_from_directory."""
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at '{TRAIN_DIR}' and '{TEST_DIR}', but one or both are missing."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    class_names = train_ds.class_names

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(class_names)} in '{TRAIN_DIR}'."
        )

    train_ds = train_ds.cache("/tmp/train_cache").prefetch(AUTOTUNE)
    test_ds = test_ds.cache("/tmp/test_cache").prefetch(AUTOTUNE)

    return train_ds, test_ds, class_names


def count_dataset_images(dataset: tf.data.Dataset) -> int:
    """Return the exact number of images in a batched dataset."""
    cardinality = tf.data.experimental.cardinality(dataset.unbatch()).numpy()
    if cardinality >= 0:
        return int(cardinality)

    return int(
        dataset.reduce(
            tf.constant(0, dtype=tf.int64),
            lambda acc, batch: acc + tf.cast(tf.shape(batch[0])[0], tf.int64),
        ).numpy()
    )


def extract_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect integer labels from a batched (images, labels) dataset."""
    all_labels: List[np.ndarray] = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())

    if not all_labels:
        return np.array([], dtype=np.int64)
    return np.concatenate(all_labels).astype(np.int64)


def save_class_names(class_names: List[str]) -> None:
    """Persist class index mapping used during training/evaluation."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def save_predictions(
    model_dir: Path,
    class_names: List[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_confidences: np.ndarray,
) -> None:
    """Save per-sample predictions for downstream confusion/error analysis."""
    if true_labels.size == 0 or pred_labels.size == 0:
        return

    if true_labels.shape[0] != pred_labels.shape[0]:
        min_len = min(true_labels.shape[0], pred_labels.shape[0], pred_confidences.shape[0])
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]
        pred_confidences = pred_confidences[:min_len]

    def label_to_name(label: int) -> str:
        if 0 <= label < len(class_names):
            return class_names[label]
        return f"class_{label}"

    output_path = model_dir / "predictions.csv"
    fieldnames = [
        "sample_index",
        "true_label",
        "pred_label",
        "true_class",
        "pred_class",
        "correct",
        "confidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (y_true, y_pred, confidence) in enumerate(
            zip(true_labels, pred_labels, pred_confidences)
        ):
            y_true_int = int(y_true)
            y_pred_int = int(y_pred)
            writer.writerow(
                {
                    "sample_index": idx,
                    "true_label": y_true_int,
                    "pred_label": y_pred_int,
                    "true_class": label_to_name(y_true_int),
                    "pred_class": label_to_name(y_pred_int),
                    "correct": int(y_true_int == y_pred_int),
                    "confidence": float(confidence),
                }
            )


def get_model(model_name: str, num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Build transfer-learning model and return (model, base_model)."""
    model_map: Dict[str, Dict[str, Callable]] = {
        "VGG16": {"base": VGG16, "preprocess": vgg16_preprocess},
        "ResNet50": {"base": ResNet50, "preprocess": resnet50_preprocess},
        "MobileNetV3Small": {
            "base": MobileNetV3Small,
            "preprocess": mobilenet_v3_preprocess,
        },
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model '{model_name}'.")

    base_constructor = model_map[model_name]["base"]
    preprocess_fn = model_map[model_name]["preprocess"]

    base_model = base_constructor(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Lambda(preprocess_fn, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # Keep output in float32 for numerical stability with mixed precision.
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs, outputs, name=f"{model_name}_classifier")
    return model, base_model


def build_callbacks(model_name: str, phase: str) -> List[callbacks.Callback]:
    """Create standard callbacks for training phases."""
    checkpoint_dir = OUTPUT_ROOT / ".checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{phase}.keras"

    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def unfreeze_top_layers(base_model: tf.keras.Model, top_layers: int = UNFREEZE_TOP_LAYERS) -> None:
    """Unfreeze top layers while keeping BatchNorm frozen for stable fine-tuning."""
    base_model.trainable = True

    if top_layers <= 0:
        for layer in base_model.layers:
            layer.trainable = False
        return

    split_idx = max(0, len(base_model.layers) - top_layers)
    for layer in base_model.layers[:split_idx]:
        layer.trainable = False

    for layer in base_model.layers[split_idx:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def train_model(model_name: str) -> Dict[str, float]:
    """Train a model with frozen-base stage then fine-tuning stage."""
    if TRAIN_DS is None or TEST_DS is None or TEST_IMAGE_COUNT is None or not CLASS_NAMES:
        raise RuntimeError("Datasets are not initialized. Call prepare_datasets() first.")

    model_dir = OUTPUT_ROOT / model_name
    checkpoint_dir = OUTPUT_ROOT / ".checkpoints" / model_name

    if model_dir.exists():
        shutil.rmtree(model_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    model, base_model = get_model(model_name, len(CLASS_NAMES))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start_time = time.perf_counter()

    history_frozen = model.fit(
        TRAIN_DS,
        validation_data=TEST_DS,
        epochs=INITIAL_EPOCHS,
        callbacks=build_callbacks(model_name, phase="frozen"),
        verbose=1,
    )

    unfreeze_top_layers(base_model, top_layers=UNFREEZE_TOP_LAYERS)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        TRAIN_DS,
        validation_data=TEST_DS,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=build_callbacks(model_name, phase="finetune"),
        verbose=1,
    )

    total_training_time = time.perf_counter() - start_time

    eval_loss, eval_acc = model.evaluate(TEST_DS, verbose=0)
    inference_start_time = time.perf_counter()
    pred_probs = model.predict(TEST_DS, verbose=0)
    total_inference_time = time.perf_counter() - inference_start_time
    inference_time_per_image = (
        total_inference_time / TEST_IMAGE_COUNT if TEST_IMAGE_COUNT > 0 else 0.0
    )
    param_count = int(model.count_params())

    pred_labels = np.argmax(pred_probs, axis=1).astype(np.int64)
    pred_confidences = np.max(pred_probs, axis=1).astype(np.float64)
    true_labels = extract_labels(TEST_DS)

    merged_history = {
        key: history_frozen.history.get(key, []) + history_finetune.history.get(key, [])
        for key in set(history_frozen.history) | set(history_finetune.history)
    }

    tf.saved_model.save(model, str(model_dir))

    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(merged_history, f, indent=2)

    with (model_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"accuracy: {eval_acc:.6f}\n")
        f.write(f"loss: {eval_loss:.6f}\n")
        f.write(f"training_time_seconds: {total_training_time:.2f}\n")
        f.write(f"parameter_count: {param_count}\n")
        f.write(f"inference_time_per_image_seconds: {inference_time_per_image:.8f}\n")
    save_predictions(
        model_dir=model_dir,
        class_names=CLASS_NAMES,
        true_labels=true_labels,
        pred_labels=pred_labels,
        pred_confidences=pred_confidences,
    )

    return {
        "model_name": model_name,
        "accuracy": float(eval_acc),
        "loss": float(eval_loss),
        "training_time_seconds": float(total_training_time),
        "parameter_count": param_count,
        "inference_time_per_image_seconds": float(inference_time_per_image),
    }


def save_comparison(results: List[Dict[str, float]]) -> None:
    """Write model comparison metrics to CSV."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_ROOT / "comparison.csv"
    fieldnames = [
        "model_name",
        "accuracy",
        "loss",
        "training_time_seconds",
        "parameter_count",
        "inference_time_per_image_seconds",
    ]

    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    global TRAIN_DS, TEST_DS, TEST_IMAGE_COUNT, CLASS_NAMES

    set_reproducibility(SEED)
    mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs: {len(gpus)}")
    if gpus:
        print(f"GPU name: {gpus[0].name}")
    else:
        print("GPU name: None")

    TRAIN_DS, TEST_DS, CLASS_NAMES = prepare_datasets()
    TEST_IMAGE_COUNT = count_dataset_images(TEST_DS)
    print(f"Detected {len(CLASS_NAMES)} classes.")
    save_class_names(CLASS_NAMES)

    model_names = ["VGG16", "ResNet50", "MobileNetV3Small"]
    all_results: List[Dict[str, float]] = []

    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        result = train_model(model_name=model_name)
        all_results.append(result)

        tf.keras.backend.clear_session()
        gc.collect()

    save_comparison(all_results)
    print("\nTraining complete. Comparison saved to /outputs/models/comparison.csv")


if __name__ == "__main__":
    main()
