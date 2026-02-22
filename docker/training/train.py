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

INITIAL_EPOCHS: int = 10
FINE_TUNE_EPOCHS: int = 10
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

    start_time = time.time()

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

    total_training_time = time.time() - start_time

    eval_loss, eval_acc = model.evaluate(TEST_DS, verbose=0)
    inference_start_time = time.time()
    model.predict(TEST_DS, verbose=0)
    total_inference_time = time.time() - inference_start_time
    inference_time_per_image = (
        total_inference_time / TEST_IMAGE_COUNT if TEST_IMAGE_COUNT > 0 else 0.0
    )
    param_count = int(model.count_params())

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
