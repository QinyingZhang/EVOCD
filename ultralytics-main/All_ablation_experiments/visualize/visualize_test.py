"""
简洁版可视化脚本
- Matplotlib 绘制，标题区展示类别/置信度/朝向
- 图中叠加组件检测框
"""

from pathlib import Path
import argparse
import csv
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 允许直接从项目根目录运行
CURRENT_DIR: Path = Path(__file__).resolve().parent
ABLATION_ROOT: Path = CURRENT_DIR.parent  # All_ablation_experiments
PROJECT_ROOT: Path = ABLATION_ROOT.parent  # 项目根目录
HIER_MULT_DIR = PROJECT_ROOT / "hier_mult"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HIER_MULT_DIR) not in sys.path:
    sys.path.insert(0, str(HIER_MULT_DIR))

from hier_mult.inference import FloodGradePredictor

ORIENTATION_DISPLAY = {
    "front": "Front",
    "rear": "Rear",
    "front_side": "Front_side",
    "rear_side": "Rear_side",
    "side": "Side",
}


def _format_header(result):
    level = result.get("flood_grade", "?")
    prob = result.get("confidence", 0.0)
    raw_orientation = result.get("orientation_name", "unknown")
    orientation = ORIENTATION_DISPLAY.get(raw_orientation, raw_orientation)
    return f"class: Level {level}   prob: {prob:.3f}   ori: {orientation}"


def visualize_simple(predictor, image_path, output_path=None, dpi=150):
    """生成示例风格的可视化结果"""
    result = predictor.predict(image_path)

    # 读取图像（RGB）
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    fig_height = max(h / 100, 3.5)
    fig_width = max(w / 100, 4.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(image)

    # 标题信息
    ax.set_title(_format_header(result), fontsize=14, pad=12)

    # 绘制组件检测框
    boxes = result.get("component_boxes")
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(float, box[:4])
            conf = round(float(box[4]), 2) if len(box) > 4 else None

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="#FFD400",
                facecolor="none",
            )
            ax.add_patch(rect)

            if conf is not None:
                ax.text(
                    x1,
                    max(y1 - 4, 0),
                    f"{conf:.2f}",
                    color="black",
                    fontsize=10,
                    bbox=dict(facecolor="#FFD400", alpha=0.8, edgecolor="none", pad=1),
                )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✅ 简洁版可视化已保存: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def _load_paths_from_csv(csv_path):
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError("CSV缺少 image_path 列")
        for row in reader:
            path = row.get("image_path", "").strip()
            if path:
                records.append(path)
    return records


def main():
    parser = argparse.ArgumentParser(description="简洁版可视化 - 适配论文小图")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(ABLATION_ROOT / "data" / "test.csv"),
        help="包含 image_path 列的 CSV（批量预测）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="单张图片自定义输出文件（批量模式请使用 --output-dir）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CURRENT_DIR / "runs"),
        help="批量输出目录（默认 visualize/runs）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="推理设备（cuda/cpu）")
    args = parser.parse_args()

    # 必须提供 image 或 csv
    if not args.image and not args.csv:
        parser.error("请至少提供 --image 或 --csv")

    # 解析图像列表
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        for p in _load_paths_from_csv(csv_path):
            image_paths.append(Path(p))

    # 检查路径
    missing = [str(p) for p in image_paths if not Path(p).exists()]
    if missing:
        missing_str = "\n".join(missing[:5])
        raise FileNotFoundError(f"以下图片不存在（仅展示前5条）:\n{missing_str}")

    # 模型
    default_checkpoint = (
        PROJECT_ROOT
        / "All_ablation_experiments"
        / "training_strategies"
        / "train"
        / "runs"
        / "progressive_training"
        / "train_01"
        / "best_model.pth"
    )

    if args.checkpoint is None:
        checkpoint_path = default_checkpoint
        print(f"ℹ️  使用默认模型: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    predictor = FloodGradePredictor(checkpoint_path=str(checkpoint_path), device=args.device)

    output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, 1):
        if args.csv or args.output is None:
            output_path = output_root / f"pre_{image_path.name}"
        else:
            output_path = Path(args.output)

        print(f"[{idx}/{total}] 处理 {image_path} -> {output_path}")
        visualize_simple(
            predictor,
            str(image_path),
            str(output_path),
        )


if __name__ == "__main__":
    main()

