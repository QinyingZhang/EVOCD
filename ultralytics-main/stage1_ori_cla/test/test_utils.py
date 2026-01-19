"""
Helper functions: Evaluate a single YOLO classification model on the specified test set.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch  # type: ignore[import]
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

try:
    from thop import profile  # type: ignore[import]

    THOP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    THOP_AVAILABLE = False
    print("⚠️  Note: thop not installed, skipping GFLOPs calculation")

# Unified class order
CLASS_NAMES: Sequence[str] = ["front", "front_side", "rear", "rear_side", "side"]
DISPLAY_LABELS = {
    "front": "Front",
    "front_side": "Front-side",
    "rear": "Rear",
    "rear_side": "Rear-side",
    "side": "Side",
}
ZERO_DIVISION_ARG: Any = 0


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    macro_precision: float
    macro_recall: float
    macro_f1_score: float
    per_class_recall: List[float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Dict[str, float]]
    report_text: str
    predictions: List[str]
    labels: List[str]
    image_paths: List[str]
    avg_inference_ms: float
    params_m: Optional[float] = None
    gflops: Optional[float] = None


def _collect_test_samples(
    test_dir: Path, class_names: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Read all image paths and labels in the test set directory."""
    images: List[str] = []
    labels: List[str] = []
    if not test_dir.exists():
        raise FileNotFoundError(f"Test set directory not found: {test_dir}")

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for class_name in class_names:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: Class directory not found - {class_dir}")
            continue
        for ext in extensions:
            for img_path in class_dir.glob(ext):
                images.append(str(img_path))
                labels.append(class_name)

    if not images:
        raise ValueError(f"No images found in test set: {test_dir}")

    print(f"Loaded {len(images)} test images")
    return images, labels


def _collect_samples_from_csv(
    csv_path: Path, class_names: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Read image paths and labels from a CSV file with an orientation column."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Test set CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"image_path", "orientation"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns: {required_cols}. Current columns: {list(df.columns)}"
        )

    images = df["image_path"].astype(str).tolist()
    labels = df["orientation"].astype(str).tolist()

    if not images:
        raise ValueError(f"No samples found in CSV: {csv_path}")

    invalid_labels = sorted({lbl for lbl in labels if lbl not in class_names})
    if invalid_labels:
        print(f"⚠️  Warning: Unknown classes in CSV: {invalid_labels}")

    print(f"Loaded {len(images)} test samples from CSV: {csv_path}")
    return images, labels


def _plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], save_path: Path, title: str
) -> None:
    """Plot and save confusion matrix."""
    tick_labels = [DISPLAY_LABELS.get(name, name) for name in class_names]
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def _save_predictions_csv(
    image_paths: List[str], labels: List[str], predictions: List[str], save_path: Path
) -> None:
    """Save per-sample prediction results."""
    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": labels,
            "predicted_label": predictions,
            "correct": [t == p for t, p in zip(labels, predictions)],
        }
    )
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Prediction details saved: {save_path}")


def _compute_metrics(
    labels: List[str],
    predictions: List[str],
    class_names: Sequence[str],
) -> Tuple[EvaluationResult, np.ndarray]:
    """Compute classification metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=ZERO_DIVISION_ARG
    )
    recall = recall_score(
        labels, predictions, average="weighted", zero_division=ZERO_DIVISION_ARG
    )
    f1_val = f1_score(
        labels, predictions, average="weighted", zero_division=ZERO_DIVISION_ARG
    )

    macro_precision = precision_score(
        labels, predictions, average="macro", zero_division=ZERO_DIVISION_ARG
    )
    macro_recall = recall_score(
        labels, predictions, average="macro", zero_division=ZERO_DIVISION_ARG
    )
    macro_f1 = f1_score(
        labels, predictions, average="macro", zero_division=ZERO_DIVISION_ARG
    )

    cm = confusion_matrix(labels, predictions, labels=class_names)
    per_class_recall: List[float] = []
    for idx, class_name in enumerate(class_names):
        total = cm[idx].sum()
        per_class_recall.append(float(cm[idx, idx] / total) if total > 0 else 0.0)

    report_dict: Dict[str, Any] = cast(
        Dict[str, Any],
        classification_report(
            labels,
            predictions,
            labels=class_names,
            target_names=class_names,
            output_dict=True,
            zero_division=ZERO_DIVISION_ARG,
            digits=4,
        ),
    )
    report_text: str = cast(
        str,
        classification_report(
        labels,
        predictions,
        labels=class_names,
        target_names=class_names,
        zero_division=ZERO_DIVISION_ARG,
        digits=4,
        ),
    )

    result = EvaluationResult(
        model_name="",
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_val),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1_score=float(macro_f1),
        per_class_recall=per_class_recall,
        confusion_matrix=cm,
        classification_report=report_dict,  # type: ignore[arg-type]
        report_text=report_text,
        predictions=predictions,
        labels=labels,
        image_paths=[],
        avg_inference_ms=0.0,
        params_m=None,
        gflops=None,
    )
    return result, cm


def _compute_model_complexity(
    model: YOLO, image_size: int
) -> Tuple[Optional[float], Optional[float]]:
    """Compute model parameters (M) and GFLOPs (if available)."""
    try:
        pytorch_model = model.model  # type: ignore[attr-defined]
        if hasattr(pytorch_model, "model"):
            pytorch_model = pytorch_model.model  # type: ignore[attr-defined]

        total_params = sum(p.numel() for p in pytorch_model.parameters())  # type: ignore[arg-type]
        params_m = total_params / 1e6

        gflops: Optional[float] = None
        if THOP_AVAILABLE:
            dummy = torch.randn(1, 3, image_size, image_size)
            flops, _ = profile(pytorch_model, inputs=(dummy,), verbose=False)  # type: ignore[arg-type]
            gflops = flops / 1e9
        return params_m, gflops
    except Exception as exc:  # pragma: no cover - fallback if calculation fails
        print(f"⚠️  Unable to compute model complexity: {exc}")
        return None, None


def run_model_test(
    *,
    model_name: str,
    model_path: Path,
    output_dir: Path,
    test_dir: Optional[Path] = None,
    samples_csv: Optional[Path] = None,
    class_names: Sequence[str] = CLASS_NAMES,
    image_size: int = 256,
) -> EvaluationResult:
    """
    Evaluate a single model and write results to the specified directory.
    Can use either a directory-structured dataset or a CSV sample list.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if samples_csv is None and test_dir is None:
        raise ValueError("You must provide either test_dir or samples_csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"Testing model: {model_name}")
    print(f"Model weights: {model_path}")
    print("=" * 80)

    if samples_csv is not None:
        images, labels = _collect_samples_from_csv(samples_csv, class_names)
        data_source_desc = f"CSV: {samples_csv}"
    else:
        assert test_dir is not None
        images, labels = _collect_test_samples(test_dir, class_names)
        data_source_desc = f"dir: {test_dir}"

    print(f"Test set: {data_source_desc}")

    print("Loading YOLO model...")
    model = YOLO(str(model_path))

    params_m, gflops = _compute_model_complexity(model, image_size)
    if params_m is not None:
        print(f"Parameters: {params_m:.2f} M")
    if gflops is not None:
        print(f"GFLOPs: {gflops:.2f}")

    predictions: List[str] = []
    inference_times: List[float] = []

    for idx, image_path in enumerate(images):
        start = time.perf_counter()
        results = model.predict(image_path, imgsz=image_size, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000.0
        inference_times.append(elapsed)

        result = results[0]
        pred_idx = int(result.probs.top1)  # type: ignore[attr-defined]
        pred_name = result.names[pred_idx]  # type: ignore[attr-defined]
        predictions.append(pred_name)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(images)} images")

    base_result, cm = _compute_metrics(labels, predictions, class_names)
    base_result.model_name = model_name
    base_result.image_paths = images
    base_result.avg_inference_ms = float(np.mean(inference_times))
    base_result.params_m = params_m
    base_result.gflops = gflops

    # Save results
    metrics_payload: Dict[str, object] = {
        "model_name": model_name,
        "num_samples": len(labels),
        "accuracy": base_result.accuracy,
        "precision": base_result.precision,
        "recall": base_result.recall,
        "f1_score": base_result.f1_score,
        "macro_precision": base_result.macro_precision,
        "macro_recall": base_result.macro_recall,
        "macro_f1_score": base_result.macro_f1_score,
        "per_class_recall": base_result.per_class_recall,
        "confusion_matrix": cm.tolist(),
        "avg_inference_ms": base_result.avg_inference_ms,
        "params_m": params_m,
        "gflops": gflops,
    }

    results_path = output_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved: {results_path}")

    report_df = pd.DataFrame(base_result.classification_report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")
    print(f"Classification report saved: {report_path}")

    predictions_path = output_dir / "predictions.csv"
    _save_predictions_csv(images, labels, predictions, predictions_path)

    cm_path = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(
        cm,
        class_names,
        cm_path,
        "Orientations Classification Confusion Matrix",
    )

    report_text_path = output_dir / "classification_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write(base_result.report_text)
    print(f"Classification report (text) saved: {report_text_path}")

    print("\n--- Metrics Summary ---")
    if params_m is not None:
        print(f"Parameters: {params_m:.2f} M")
    if gflops is not None:
        print(f"GFLOPs: {gflops:.2f}")
    print(f"Average inference time: {base_result.avg_inference_ms:.2f} ms/img")
    print(f"Accuracy: {base_result.accuracy*100:.2f}%")
    print(f"Macro F1: {base_result.macro_f1_score*100:.2f}%")
    print(f"Test complete, output dir: {output_dir}")

    return base_result

