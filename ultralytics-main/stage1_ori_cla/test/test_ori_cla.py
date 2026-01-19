
import argparse
from pathlib import Path

from test_utils import CLASS_NAMES, run_model_test

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "stage1_ori_cla"
    / "train"
    / "runs"
    / "train_01"
    / "weights"
    / "best.pt"
)
DEFAULT_TEST_DIR = (
    PROJECT_ROOT / "stage1_ori_cla" / "dataset" / "test"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs" / "train_01"
DEFAULT_TEST_CSV = (
    PROJECT_ROOT / "All_ablation_experiments" / "data" / "test.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orientation classification for default n model (YOLOv11n-cls)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to best.pt model file",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        help=(
            "Test set root directory (contains front etc. subdirectories), leave empty to use CSV, "
            f"Example: {DEFAULT_TEST_DIR}"
        ),
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
        help="Test set CSV containing image_path and orientation columns."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for test results",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size for inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model_test(
        model_name="YOLOv11n-cls (n 模型)",
        model_path=Path(args.model_path),
        test_dir=Path(args.test_dir) if args.test_dir else None,
        samples_csv=Path(args.test_csv) if args.test_csv else None,
        output_dir=Path(args.output_dir),
        class_names=CLASS_NAMES,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()

