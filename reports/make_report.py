#!/usr/bin/env python3
"""Generate thesis report artifacts from training outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MODELS_DIR = REPO_ROOT / "outputs" / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_CSV = REPORTS_DIR / "summary.csv"
ERROR_ANALYSIS_CSV = REPORTS_DIR / "error_analysis.csv"
TOP_CONFUSIONS_CSV = REPORTS_DIR / "top_confusions.csv"

SUMMARY_COLUMNS = [
    "model_name",
    "accuracy",
    "loss",
    "training_time_seconds",
    "parameter_count",
    "inference_time_per_image_seconds",
]


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def _to_float(value: object, field_name: str, model_name: str, source: Path) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        warn(f"Missing '{field_name}' for model '{model_name}' in '{source}'.")
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        warn(
            f"Could not parse '{field_name}' value '{value}' for model '{model_name}' in '{source}'."
        )
        return float("nan")


def _to_int(value: object, field_name: str, model_name: str, source: Path) -> float:
    number = _to_float(value, field_name, model_name, source)
    if np.isnan(number):
        return float("nan")
    return int(number)


def normalize_metrics_row(raw_row: Dict[str, object], source: Path) -> Dict[str, object]:
    model_name = str(raw_row.get("model_name", "")).strip()
    if not model_name:
        model_name = "unknown_model" if source.name == "comparison.csv" else source.parent.name
        warn(f"'model_name' missing in '{source}', using '{model_name}'.")

    accuracy = _to_float(raw_row.get("accuracy"), "accuracy", model_name, source)
    loss = _to_float(raw_row.get("loss"), "loss", model_name, source)
    training_time_seconds = _to_float(
        raw_row.get("training_time_seconds"), "training_time_seconds", model_name, source
    )
    parameter_count = _to_int(raw_row.get("parameter_count"), "parameter_count", model_name, source)

    raw_inference_seconds = raw_row.get("inference_time_per_image_seconds")
    raw_inference_ms = raw_row.get("inference_time_ms")

    inference_seconds = float("nan")
    inference_ms = float("nan")
    inference_from_ms = False

    if raw_inference_seconds is not None and not (
        isinstance(raw_inference_seconds, float) and np.isnan(raw_inference_seconds)
    ):
        inference_seconds = _to_float(
            raw_inference_seconds, "inference_time_per_image_seconds", model_name, source
        )
    elif raw_inference_ms is not None and not (
        isinstance(raw_inference_ms, float) and np.isnan(raw_inference_ms)
    ):
        inference_ms = _to_float(raw_inference_ms, "inference_time_ms", model_name, source)
        if not np.isnan(inference_ms):
            inference_seconds = inference_ms / 1000.0
            inference_from_ms = True
    else:
        warn(
            "Missing inference timing for model "
            f"'{model_name}' in '{source}'. Expected "
            "'inference_time_per_image_seconds' or 'inference_time_ms'."
        )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "loss": loss,
        "training_time_seconds": training_time_seconds,
        "parameter_count": parameter_count,
        "inference_time_per_image_seconds": inference_seconds,
        "inference_time_ms": inference_ms,
        "_inference_from_ms": inference_from_ms,
    }


def parse_metrics_file(metrics_path: Path) -> Optional[Dict[str, object]]:
    if not metrics_path.exists():
        warn(f"Missing metrics file: '{metrics_path}'.")
        return None

    raw: Dict[str, object] = {}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                if ":" not in text:
                    warn(
                        f"Skipping malformed line {line_number} in '{metrics_path}': '{text}'."
                    )
                    continue
                key, value = text.split(":", maxsplit=1)
                raw[key.strip()] = value.strip()
    except OSError as exc:
        warn(f"Failed reading metrics file '{metrics_path}': {exc}")
        return None

    if "model_name" not in raw or not str(raw["model_name"]).strip():
        raw["model_name"] = metrics_path.parent.name
    return normalize_metrics_row(raw, metrics_path)


def model_directories() -> List[Path]:
    if not OUTPUT_MODELS_DIR.exists():
        warn(f"Outputs folder not found: '{OUTPUT_MODELS_DIR}'.")
        return []

    directories = [
        path
        for path in sorted(OUTPUT_MODELS_DIR.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
    if not directories:
        warn(f"No model directories found in '{OUTPUT_MODELS_DIR}'.")
    return directories


def load_metrics_from_comparison(comparison_path: Path) -> pd.DataFrame:
    try:
        raw_df = pd.read_csv(comparison_path)
    except Exception as exc:  # noqa: BLE001
        warn(f"Failed reading comparison CSV '{comparison_path}': {exc}")
        return pd.DataFrame()

    if raw_df.empty:
        warn(f"Comparison CSV is empty: '{comparison_path}'.")
        return pd.DataFrame()

    rows = [
        normalize_metrics_row(row, comparison_path) for row in raw_df.to_dict(orient="records")
    ]
    return pd.DataFrame(rows)


def load_metrics_from_files(directories: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_dir in directories:
        parsed = parse_metrics_file(model_dir / "metrics.txt")
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        warn(
            "No usable metrics found from model folders. "
            "Expected files like './outputs/models/<MODEL>/metrics.txt'."
        )
        return pd.DataFrame(columns=SUMMARY_COLUMNS + ["inference_time_ms", "_inference_from_ms"])

    return pd.DataFrame(rows)


def load_metrics() -> pd.DataFrame:
    comparison_path = OUTPUT_MODELS_DIR / "comparison.csv"
    directories = model_directories()

    if comparison_path.exists():
        info(f"Loading metrics from '{comparison_path}'.")
        comparison_df = load_metrics_from_comparison(comparison_path)
        if not comparison_df.empty:
            return comparison_df
        warn("Falling back to per-model metrics files because comparison.csv could not be used.")
    else:
        warn(
            f"Comparison CSV not found at '{comparison_path}'. "
            "Falling back to per-model metrics files."
        )

    return load_metrics_from_files(directories)


def write_summary(summary_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(SUMMARY_CSV, index=False)
        warn(f"No summary rows available. Wrote header-only CSV to '{SUMMARY_CSV}'.")
        return

    export_df = summary_df[SUMMARY_COLUMNS].copy()
    export_df = export_df.sort_values(by="model_name").reset_index(drop=True)
    export_df.to_csv(SUMMARY_CSV, index=False)
    info(f"Wrote summary CSV to '{SUMMARY_CSV}'.")


def plot_bar_chart(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if df.empty:
        warn(f"Skipping '{output_path.name}' because there is no summary data.")
        return

    valid = df[["model_name", column]].dropna()
    if valid.empty:
        warn(f"Skipping '{output_path.name}' because '{column}' has no valid values.")
        return

    names = valid["model_name"].astype(str).tolist()
    values = valid[column].astype(float).to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, 1.5 * len(names)), 4.5))
    plt.bar(names, values)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    info(f"Wrote figure '{output_path}'.")


def plot_aggregate_figures(summary_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_bar_chart(
        summary_df,
        column="accuracy",
        ylabel="Accuracy",
        title="Accuracy by Model",
        output_path=FIGURES_DIR / "accuracy_by_model.png",
    )

    all_from_ms = (
        not summary_df.empty
        and summary_df["_inference_from_ms"].fillna(False).astype(bool).all()
        and summary_df["inference_time_ms"].notna().any()
    )
    if all_from_ms:
        inference_column = "inference_time_ms"
        inference_ylabel = "Inference Time (ms / image)"
        inference_title = "Inference Time by Model (ms)"
    else:
        inference_column = "inference_time_per_image_seconds"
        inference_ylabel = "Inference Time (seconds / image)"
        inference_title = "Inference Time by Model (seconds)"

    plot_bar_chart(
        summary_df,
        column=inference_column,
        ylabel=inference_ylabel,
        title=inference_title,
        output_path=FIGURES_DIR / "inference_time_by_model.png",
    )

    plot_bar_chart(
        summary_df,
        column="parameter_count",
        ylabel="Parameter Count",
        title="Parameter Count by Model",
        output_path=FIGURES_DIR / "params_by_model.png",
    )


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _to_numeric_array(values: object, model_name: str, metric_name: str, history_path: Path) -> np.ndarray:
    if values is None:
        return np.array([])
    if not isinstance(values, list):
        warn(
            f"History key '{metric_name}' for '{model_name}' in '{history_path}' is not a list. "
            "Skipping this series."
        )
        return np.array([])

    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        warn(
            f"Could not parse history key '{metric_name}' as numeric values for "
            f"'{model_name}' in '{history_path}'."
        )
        return np.array([])


def plot_learning_curve(
    model_name: str,
    train_values: np.ndarray,
    val_values: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if train_values.size == 0 and val_values.size == 0:
        warn(f"Skipping '{output_path.name}' for '{model_name}' because no curve data is available.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.0))
    if train_values.size > 0:
        plt.plot(np.arange(1, train_values.size + 1), train_values, label="train")
    if val_values.size > 0:
        plt.plot(np.arange(1, val_values.size + 1), val_values, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    info(f"Wrote figure '{output_path}'.")


def generate_learning_curves(model_dirs: Iterable[Path]) -> None:
    for model_dir in model_dirs:
        model_name = model_dir.name
        history_path = model_dir / "history.json"
        if not history_path.exists():
            warn(f"Missing history file for '{model_name}': '{history_path}'.")
            continue

        try:
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            warn(f"Failed reading history for '{model_name}' from '{history_path}': {exc}")
            continue

        if not isinstance(history, dict):
            warn(f"History for '{model_name}' in '{history_path}' is not a JSON object.")
            continue

        model_figures_dir = FIGURES_DIR / _sanitize_name(model_name)

        train_acc = _to_numeric_array(
            history.get("accuracy", history.get("acc")),
            model_name,
            "accuracy",
            history_path,
        )
        val_acc = _to_numeric_array(
            history.get("val_accuracy", history.get("val_acc")),
            model_name,
            "val_accuracy",
            history_path,
        )
        plot_learning_curve(
            model_name=model_name,
            train_values=train_acc,
            val_values=val_acc,
            ylabel="Accuracy",
            title=f"{model_name} Accuracy Curve",
            output_path=model_figures_dir / "accuracy_curve.png",
        )

        train_loss = _to_numeric_array(history.get("loss"), model_name, "loss", history_path)
        val_loss = _to_numeric_array(history.get("val_loss"), model_name, "val_loss", history_path)
        plot_learning_curve(
            model_name=model_name,
            train_values=train_loss,
            val_values=val_loss,
            ylabel="Loss",
            title=f"{model_name} Loss Curve",
            output_path=model_figures_dir / "loss_curve.png",
        )


def load_class_names() -> List[str]:
    class_names_path = OUTPUT_MODELS_DIR / "class_names.json"
    if not class_names_path.exists():
        warn(
            f"Missing class name mapping: '{class_names_path}'. "
            "Class labels will use fallback names where needed."
        )
        return []

    try:
        with class_names_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"Failed reading class name mapping from '{class_names_path}': {exc}")
        return []

    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        warn(f"Invalid class name mapping in '{class_names_path}'. Expected a JSON string list.")
        return []

    return data


def load_predictions(model_dir: Path) -> Optional[pd.DataFrame]:
    predictions_path = model_dir / "predictions.csv"
    if not predictions_path.exists():
        warn(f"Missing predictions file for '{model_dir.name}': '{predictions_path}'.")
        return None

    try:
        df = pd.read_csv(predictions_path)
    except Exception as exc:  # noqa: BLE001
        warn(f"Failed reading predictions file '{predictions_path}': {exc}")
        return None

    required_columns = {"true_label", "pred_label"}
    missing = required_columns.difference(df.columns)
    if missing:
        warn(
            f"Predictions file '{predictions_path}' is missing required columns: "
            f"{sorted(missing)}."
        )
        return None

    df = df.copy()
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce")
    valid_mask = df["true_label"].notna() & df["pred_label"].notna()
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        warn(
            f"Dropping {dropped} invalid prediction rows in '{predictions_path}' "
            "because labels are not numeric."
        )

    cleaned = df.loc[valid_mask].copy()
    cleaned["true_label"] = cleaned["true_label"].astype(int)
    cleaned["pred_label"] = cleaned["pred_label"].astype(int)
    if cleaned.empty:
        warn(f"No valid prediction rows in '{predictions_path}'.")
        return None

    return cleaned


def build_label_space(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: Sequence[str],
) -> List[int]:
    labels = set(np.unique(true_labels).tolist()) | set(np.unique(pred_labels).tolist())
    if class_names:
        labels |= set(range(len(class_names)))
    return sorted(int(label) for label in labels)


def label_name(label: int, class_names: Sequence[str]) -> str:
    if 0 <= label < len(class_names):
        return class_names[label]
    return f"class_{label}"


def compute_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    labels: Sequence[int],
) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for y_true, y_pred in zip(true_labels, pred_labels):
        true_idx = label_to_index.get(int(y_true))
        pred_idx = label_to_index.get(int(y_pred))
        if true_idx is None or pred_idx is None:
            continue
        matrix[true_idx, pred_idx] += 1

    return matrix


def save_confusion_matrix_tables(
    model_name: str,
    labels: Sequence[int],
    class_names: Sequence[str],
    confusion_matrix: np.ndarray,
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    names = [f"{label}:{label_name(label, class_names)}" for label in labels]

    cm_df = pd.DataFrame(confusion_matrix, index=names, columns=names)
    cm_path = TABLES_DIR / f"{_sanitize_name(model_name)}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    info(f"Wrote table '{cm_path}'.")

    row_totals = confusion_matrix.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        confusion_matrix.astype(np.float64),
        row_totals,
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    cm_norm_df = pd.DataFrame(cm_norm, index=names, columns=names)
    cm_norm_path = TABLES_DIR / f"{_sanitize_name(model_name)}_confusion_matrix_normalized.csv"
    cm_norm_df.to_csv(cm_norm_path)
    info(f"Wrote table '{cm_norm_path}'.")


def plot_confusion_matrix(
    model_name: str,
    labels: Sequence[int],
    class_names: Sequence[str],
    confusion_matrix: np.ndarray,
    normalized: bool,
    output_path: Path,
) -> None:
    if confusion_matrix.size == 0:
        warn(f"Skipping confusion matrix for '{model_name}' because matrix is empty.")
        return

    if normalized:
        row_totals = confusion_matrix.sum(axis=1, keepdims=True)
        matrix_to_plot = np.divide(
            confusion_matrix.astype(np.float64),
            row_totals,
            out=np.zeros_like(confusion_matrix, dtype=np.float64),
            where=row_totals != 0,
        )
        colorbar_label = "Row-normalized ratio"
        title = f"{model_name} Confusion Matrix (Normalized)"
    else:
        matrix_to_plot = confusion_matrix
        colorbar_label = "Count"
        title = f"{model_name} Confusion Matrix"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = max(8.0, min(14.0, 0.28 * len(labels) + 6.0))
    plt.figure(figsize=(size, size))
    plt.imshow(matrix_to_plot, interpolation="nearest", cmap="Blues")
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    tick_positions = np.arange(len(labels))
    if len(labels) <= 25:
        tick_labels = [label_name(label, class_names) for label in labels]
    else:
        tick_labels = [str(label) for label in labels]

    plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=8)
    plt.yticks(tick_positions, tick_labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    info(f"Wrote figure '{output_path}'.")


def per_class_error_rows(
    model_name: str,
    labels: Sequence[int],
    class_names: Sequence[str],
    confusion_matrix: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    supports = confusion_matrix.sum(axis=1)
    predicted_totals = confusion_matrix.sum(axis=0)

    for idx, label in enumerate(labels):
        tp = int(confusion_matrix[idx, idx])
        support = int(supports[idx])
        fp = int(predicted_totals[idx] - tp)
        fn = int(support - tp)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / support) if support > 0 else float("nan")
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0.0
            else float("nan")
        )
        misclassification_rate = float(fn / support) if support > 0 else float("nan")

        row_without_self = confusion_matrix[idx].copy()
        if row_without_self.size > 0:
            row_without_self[idx] = 0
        top_conf_idx = int(np.argmax(row_without_self)) if row_without_self.size > 0 else -1
        top_conf_count = int(row_without_self[top_conf_idx]) if top_conf_idx >= 0 else 0
        top_conf_label = int(labels[top_conf_idx]) if top_conf_idx >= 0 and top_conf_count > 0 else np.nan
        top_conf_name = (
            label_name(int(top_conf_label), class_names) if not np.isnan(top_conf_label) else ""
        )

        rows.append(
            {
                "model_name": model_name,
                "class_label": int(label),
                "class_name": label_name(int(label), class_names),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "misclassification_rate": misclassification_rate,
                "top_confused_with_label": top_conf_label,
                "top_confused_with_class": top_conf_name,
                "top_confusion_count": top_conf_count,
            }
        )

    return rows


def top_confusion_rows(
    model_name: str,
    labels: Sequence[int],
    class_names: Sequence[str],
    confusion_matrix: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(confusion_matrix[i, j])
            if count <= 0:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "true_label": int(true_label),
                    "true_class": label_name(int(true_label), class_names),
                    "pred_label": int(pred_label),
                    "pred_class": label_name(int(pred_label), class_names),
                    "count": count,
                }
            )
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows


def generate_confusion_and_error_analysis(model_dirs: Iterable[Path]) -> None:
    class_names = load_class_names()
    error_rows: List[Dict[str, object]] = []
    top_conf_rows: List[Dict[str, object]] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        predictions = load_predictions(model_dir)
        if predictions is None:
            continue

        true_labels = predictions["true_label"].to_numpy(dtype=np.int64)
        pred_labels = predictions["pred_label"].to_numpy(dtype=np.int64)
        labels = build_label_space(true_labels, pred_labels, class_names)
        if not labels:
            warn(f"Skipping confusion/error analysis for '{model_name}' because label space is empty.")
            continue

        cm = compute_confusion_matrix(true_labels, pred_labels, labels)
        model_figures_dir = FIGURES_DIR / _sanitize_name(model_name)

        plot_confusion_matrix(
            model_name=model_name,
            labels=labels,
            class_names=class_names,
            confusion_matrix=cm,
            normalized=False,
            output_path=model_figures_dir / "confusion_matrix.png",
        )
        plot_confusion_matrix(
            model_name=model_name,
            labels=labels,
            class_names=class_names,
            confusion_matrix=cm,
            normalized=True,
            output_path=model_figures_dir / "confusion_matrix_normalized.png",
        )
        save_confusion_matrix_tables(
            model_name=model_name,
            labels=labels,
            class_names=class_names,
            confusion_matrix=cm,
        )

        error_rows.extend(
            per_class_error_rows(
                model_name=model_name,
                labels=labels,
                class_names=class_names,
                confusion_matrix=cm,
            )
        )
        top_conf_rows.extend(
            top_confusion_rows(
                model_name=model_name,
                labels=labels,
                class_names=class_names,
                confusion_matrix=cm,
            )
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if error_rows:
        error_df = pd.DataFrame(error_rows)
        error_df = error_df.sort_values(
            by=["model_name", "misclassification_rate", "support"],
            ascending=[True, False, False],
        )
        error_df.to_csv(ERROR_ANALYSIS_CSV, index=False)
        info(f"Wrote error analysis CSV to '{ERROR_ANALYSIS_CSV}'.")
    else:
        pd.DataFrame(
            columns=[
                "model_name",
                "class_label",
                "class_name",
                "support",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "misclassification_rate",
                "top_confused_with_label",
                "top_confused_with_class",
                "top_confusion_count",
            ]
        ).to_csv(ERROR_ANALYSIS_CSV, index=False)
        warn(f"No prediction data found. Wrote header-only CSV to '{ERROR_ANALYSIS_CSV}'.")

    if top_conf_rows:
        top_conf_df = pd.DataFrame(top_conf_rows)
        top_conf_df = top_conf_df.sort_values(by=["model_name", "count"], ascending=[True, False])
        top_conf_df.to_csv(TOP_CONFUSIONS_CSV, index=False)
        info(f"Wrote top confusion pairs CSV to '{TOP_CONFUSIONS_CSV}'.")
    else:
        pd.DataFrame(
            columns=[
                "model_name",
                "true_label",
                "true_class",
                "pred_label",
                "pred_class",
                "count",
            ]
        ).to_csv(TOP_CONFUSIONS_CSV, index=False)
        warn(f"No confusion pairs found. Wrote header-only CSV to '{TOP_CONFUSIONS_CSV}'.")


def main() -> None:
    info(f"Repository root: '{REPO_ROOT}'")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = model_directories()
    summary_df = load_metrics()
    write_summary(summary_df)
    plot_aggregate_figures(summary_df)
    generate_learning_curves(model_dirs)
    generate_confusion_and_error_analysis(model_dirs)

    info("Report generation complete.")


if __name__ == "__main__":
    main()
