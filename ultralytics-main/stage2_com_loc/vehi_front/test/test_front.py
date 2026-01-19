"""共享的前向检测测试脚本逻辑."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
VEHI_FRONT_DIR = SCRIPT_DIR.parent
DATA_YAML = VEHI_FRONT_DIR / "front_d.yaml"
TEST_IMAGES_DIR = VEHI_FRONT_DIR / "dataset" / "images" / "test"
DEFAULT_PROJECT_DIR = SCRIPT_DIR / "runs"


def _build_parser(default_weights: Path, default_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv11 detector on the front-orientation test split."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to the trained weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML),
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size used during validation.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS/mAP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_PROJECT_DIR),
        help="Directory where YOLO stores evaluation outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=default_name,
        help="Name of the evaluation run (sub-folder under project).",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="Save predictions to *.txt files.",
    )
    parser.add_argument(
        "--save_conf",
        action="store_true",
        help="Save confidences with --save_txt.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save COCO-style JSON results.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save evaluation plots (PR curves, confusion matrix, etc.).",
    )
    return parser


def _validate_paths(weights_path: Path) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data config not found: {DATA_YAML}")
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test image directory not found: {TEST_IMAGES_DIR}")


def _save_metrics_json(output_dir: Path, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "test_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {json_path}")


def run_front_test(default_weights: Path, default_name: str) -> None:
    parser = _build_parser(default_weights, default_name)
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()
    run_output = project_dir / args.name

    _validate_paths(weights_path)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_json=args.save_json,
        plots=args.plots,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
    )

    metrics_summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "test_images": len(list(TEST_IMAGES_DIR.glob("*"))),
    }
    _save_metrics_json(run_output, metrics_summary)
    print(
        f"mAP50: {metrics_summary['map50']:.4f}, "
        f"mAP50-95: {metrics_summary['map50_95']:.4f}, "
        f"P: {metrics_summary['precision']:.4f}, "
        f"R: {metrics_summary['recall']:.4f}"
    )


__all__ = ["run_front_test"]

if __name__ == "__main__":
    import argparse
    parser = _build_parser(
        default_weights=Path("./runs/train_n_01/weights/best.pt"),
        default_name="test_n_01"
    )
    args = parser.parse_args()
    run_front_test(args)

































