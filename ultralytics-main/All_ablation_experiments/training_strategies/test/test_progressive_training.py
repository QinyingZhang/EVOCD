"""
训练策略消融测试：完全分阶段训练（Progressive Training）
加载 progressive_training 的最佳模型并在测试集上评估性能。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SCRIPT_DIR = Path(__file__).parent
# test -> training_strategies -> All_ablation_experiments -> ultralytics-main
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
HIER_MULT_DIR = PROJECT_ROOT / "hier_mult"

if str(HIER_MULT_DIR) not in sys.path:
    sys.path.insert(0, str(HIER_MULT_DIR))

from archived.model_concat_fusion import HierarchicalFloodClassifier_ConcatFusion  # type: ignore
from dataset import FloodDataset, create_dataloader  # type: ignore
from settings import COMPONENT_MODELS, NUM_FLOOD_GRADES, ORIENTATION_MODEL  # type: ignore

ZERO_DIVISION_VALUE: Any = 0


def build_test_loader(args) -> torch.utils.data.DataLoader:
    """构建测试集数据加载器"""
    test_dataset = FloodDataset(
        csv_path=args.test_csv,
        image_size=args.image_size,
        is_train=False,
        augment=False,
    )
    loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"测试批次数:   {len(loader)}")
    return loader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    """评估模型并返回指标"""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["flood_grade"].to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs.get("flood_grade")
                if logits is None:
                    raise KeyError("模型输出中缺少 flood_grade 键")
            else:
                logits = outputs

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1} 个批次")

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    precision = precision_score(
        all_labels_np,
        all_preds_np,
        average="weighted",
        zero_division=ZERO_DIVISION_VALUE,
    )
    recall = recall_score(
        all_labels_np,
        all_preds_np,
        average="weighted",
        zero_division=ZERO_DIVISION_VALUE,
    )
    f1 = f1_score(
        all_labels_np,
        all_preds_np,
        average="weighted",
        zero_division=ZERO_DIVISION_VALUE,
    )
    macro_precision = precision_score(
        all_labels_np,
        all_preds_np,
        average="macro",
        zero_division=ZERO_DIVISION_VALUE,
    )
    macro_recall = recall_score(
        all_labels_np,
        all_preds_np,
        average="macro",
        zero_division=ZERO_DIVISION_VALUE,
    )
    macro_f1 = f1_score(
        all_labels_np,
        all_preds_np,
        average="macro",
        zero_division=ZERO_DIVISION_VALUE,
    )

    cm = confusion_matrix(
        all_labels_np, all_preds_np, labels=list(range(num_classes))
    )
    per_class_recall: List[float] = []
    for class_idx in range(num_classes):
        class_total = cm[class_idx].sum()
        if class_total > 0:
            per_class_recall.append(float(cm[class_idx, class_idx] / class_total))
        else:
            per_class_recall.append(float(ZERO_DIVISION_VALUE))

    class_report = classification_report(
        all_labels_np,
        all_preds_np,
        labels=list(range(num_classes)),
        zero_division=ZERO_DIVISION_VALUE,
        digits=4,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "predictions": all_preds,
        "labels": all_labels,
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, num_classes: int) -> None:
    """绘制混淆矩阵"""
    label_names = [f"Level {i+1}" for i in range(num_classes)]  # 显示值 1-5
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 混淆矩阵已保存: {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("训练策略消融测试：完全分阶段训练（Progressive Training）")
    print("=" * 70)

    DATA_DIR = PROJECT_ROOT / "All_ablation_experiments" / "data"
    TRAIN_RUN_DIR = (
        PROJECT_ROOT
        / "All_ablation_experiments"
        / "training_strategies"
        / "train"
        / "runs"
        / "progressive_training"
        / "train_03"
    )

    parser = argparse.ArgumentParser(description="测试训练策略消融：完全分阶段训练")
    parser.add_argument("--train_csv", type=str, default=str(DATA_DIR / "train.csv"))
    parser.add_argument("--val_csv", type=str, default=str(DATA_DIR / "val.csv"))
    parser.add_argument("--test_csv", type=str, default=str(DATA_DIR / "test.csv"))
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(TRAIN_RUN_DIR / "best_model.pth"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "All_ablation_experiments"
            / "training_strategies"
            / "test"
            / "runs"
            / "progressive_training"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 输出目录: {output_dir}")
    print(f"📊 测试数据: {args.test_csv}")
    print(f"🤖 模型路径: {args.model_path}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_loader = build_test_loader(args)

    component_paths = {k: str(v) for k, v in COMPONENT_MODELS.items()}
    model = HierarchicalFloodClassifier_ConcatFusion(
        orientation_model_path=str(ORIENTATION_MODEL),
        component_model_paths=component_paths,
        num_flood_classes=NUM_FLOOD_GRADES,
        freeze_backbone=False,
        use_component_branch=True,
    ).to(device)

    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ 模型已加载: {checkpoint_path}")

    print("\n" + "=" * 70)
    print("开始评估模型...")
    print("=" * 70)

    metrics = evaluate_model(model, test_loader, device, NUM_FLOOD_GRADES)

    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):  {metrics['f1_score']:.4f}")
    print(f"宏平均精确率 (Macro Precision): {metrics['macro_precision']:.4f}")
    print(f"宏平均召回率 (Macro Recall):     {metrics['macro_recall']:.4f}")
    print(f"宏平均F1 (Macro F1-Score):      {metrics['macro_f1_score']:.4f}")

    print("\n各等级召回率 (Per-Class Recall):")
    for class_idx, recall_value in enumerate(metrics["per_class_recall"]):
        print(f"  Level {class_idx+1}: {recall_value:.4f}")

    print("\n分类报告:")
    print(metrics["classification_report"])

    results_file = output_dir / "test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1_score": metrics["macro_f1_score"],
                "per_class_recall": metrics["per_class_recall"],
                "classification_report": metrics["classification_report"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n✅ 评估结果已保存: {results_file}")

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        cm_path,
        NUM_FLOOD_GRADES,
    )

    print("\n" + "=" * 70)
    print("训练策略消融（完全分阶段训练）测试完成")
    print("=" * 70)

