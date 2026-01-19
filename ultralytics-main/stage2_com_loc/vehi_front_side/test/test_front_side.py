"""前侧视角部件检测的通用测试脚本."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
VEHI_FRONT_SIDE_DIR = SCRIPT_DIR.parent
DATA_YAML = VEHI_FRONT_SIDE_DIR / "front_side_d.yaml"
TEST_IMAGES_DIR = VEHI_FRONT_SIDE_DIR / "dataset" / "images" / "test"
DEFAULT_PROJECT_DIR = SCRIPT_DIR / "runs"


def _build_parser(default_weights: Path, default_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO models on the front-side component detection test split."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to trained weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML),
        help="Dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size for validation.",
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
        help="IoU threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_PROJECT_DIR),
        help="Directory where YOLO stores evaluation artifacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=default_name,
        help="Sub-folder name under project dir.",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="Save predictions to label files.",
    )
    parser.add_argument(
        "--save_conf",
        action="store_true",
        help="Save confidences alongside --save_txt.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Export COCO-format JSON results.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save PR curves, confusion matrix, etc.",
    )
    return parser


def _validate_inputs(weights_path: Path) -> None:
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


def run_front_side_test(default_weights: Path, default_name: str) -> None:
    parser = _build_parser(default_weights, default_name)
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()
    run_output = project_dir / args.name

    _validate_inputs(weights_path)

    model = YOLO(str(weights_path))
    results = model.val(
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
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "test_images": len(list(TEST_IMAGES_DIR.glob("*"))),
    }
    _save_metrics_json(run_output, metrics_summary)
    print(
        "Test finished - "
        f"mAP50: {metrics_summary['map50']:.4f}, "
        f"mAP50-95: {metrics_summary['map50_95']:.4f}, "
        f"P: {metrics_summary['precision']:.4f}, "
        f"R: {metrics_summary['recall']:.4f}"
    )


__all__ = ["run_front_side_test"]

if __name__ == "__main__":
    import argparse
    parser = _build_parser(
        default_weights=Path("./runs/train_n_01/weights/best.pt"),
        default_name="test_n_01"
    )
    args = parser.parse_args()
    run_front_side_test(args.weights, args.name)

































