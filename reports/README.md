# Reports

Generate thesis-ready summary tables and figures from training outputs.

## Run

From the repository root:

```bash
python reports/make_report.py
```

## Expected Inputs

The script reads model outputs from:

- `./outputs/models/comparison.csv` (preferred, if present)
- `./outputs/models/class_names.json` (optional but recommended for readable class labels)
- `./outputs/models/<MODEL_NAME>/metrics.txt` (fallback)
- `./outputs/models/<MODEL_NAME>/history.json` (optional, for learning curves)
- `./outputs/models/<MODEL_NAME>/predictions.csv` (optional, for confusion matrix + error analysis)
- `./outputs/models/.checkpoints/<MODEL_NAME>/best_*.keras` (optional, not required for report generation)

## Produced Outputs

- `./reports/summary.csv`
  - Columns: `model_name, accuracy, loss, training_time_seconds, parameter_count, inference_time_per_image_seconds`
- Aggregate figures in `./reports/figures/`:
  - `accuracy_by_model.png`
  - `inference_time_by_model.png`
  - `params_by_model.png`
- Per-model learning curves when history is available:
  - `./reports/figures/<MODEL>/accuracy_curve.png`
  - `./reports/figures/<MODEL>/loss_curve.png`
- Section 6.2 artifacts (when predictions are available):
  - `./reports/figures/<MODEL>/confusion_matrix.png`
  - `./reports/figures/<MODEL>/confusion_matrix_normalized.png`
  - `./reports/tables/<MODEL>_confusion_matrix.csv`
  - `./reports/tables/<MODEL>_confusion_matrix_normalized.csv`
  - `./reports/error_analysis.csv` (per-class precision/recall/F1/error metrics)
  - `./reports/top_confusions.csv` (most frequent off-diagonal confusion pairs)

## Notes

- The script is resilient to missing files: it prints warnings and continues.
- If no usable metrics are found, `summary.csv` is still created with header columns.
- If prediction files are missing, confusion/error-analysis CSVs are still created with headers.
