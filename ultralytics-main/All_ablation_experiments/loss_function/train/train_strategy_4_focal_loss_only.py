"""
损失函数消融实验4：仅Focal Loss（无类别权重，无过采样，无数据增强）
使用简单拼接融合模型
训练策略：完全分阶段训练（不冻结）
"""
import argparse
import sys
from pathlib import Path
from typing import cast
import torch
import torch.nn as nn

# 添加hier_mult目录到Python路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
HIER_MULT_DIR = PROJECT_ROOT / 'hier_mult'

# 确保hier_mult目录在Python路径中
if str(HIER_MULT_DIR) not in sys.path:
    sys.path.insert(0, str(HIER_MULT_DIR))

# 导入hier_mult模块
from archived.model_concat_fusion import HierarchicalFloodClassifier_ConcatFusion  # type: ignore
from settings import ORIENTATION_MODEL, COMPONENT_MODELS, NUM_FLOOD_GRADES  # type: ignore
from train import Trainer  # type: ignore


if __name__ == '__main__':
    print("=" * 70)
    print("训练策略消融实验4：仅Focal Loss配置")
    print("Focal Loss (γ=1.5) | 无类别权重 | 无过采样 | 无数据增强")
    print("=" * 70)
    
    # 获取数据集目录（All_ablation_experiments/data）
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'All_ablation_experiments' / 'data'
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练策略消融：仅Focal Loss')
    # 数据参数
    parser.add_argument('--train_csv', type=str, default=str(DATA_DIR / 'train.csv'))
    parser.add_argument('--val_csv', type=str, default=str(DATA_DIR / 'val.csv'))
    parser.add_argument('--test_csv', type=str, default=str(DATA_DIR / 'test.csv'))
    parser.add_argument('--image_size', type=int, default=640)
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    # 模型参数
    parser.add_argument('--freeze_backbone', type=bool, default=False)
    parser.add_argument('--use_component_branch', type=bool, default=True)
    # 数据增强 - 关闭
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--oversample', type=bool, default=False)
    parser.add_argument('--oversample_level0', type=int, default=1)
    parser.add_argument('--oversample_level1', type=int, default=1)
    # 损失函数 - 仅Focal Loss
    parser.add_argument('--use_class_weights', type=bool, default=False)  # 不使用类别权重
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--use_focal_loss', type=bool, default=True)  # 开启Focal Loss
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    # 训练策略
    parser.add_argument('--mixed_precision', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    # 日志和保存
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs_dir', type=str, default=str(SCRIPT_DIR / 'runs' / 'strategy_4_focal_loss_only'))
    args = parser.parse_args()
    # 创建训练器
    trainer = Trainer(args)
    # 替换模型为消融实验版本（简单拼接融合）
    print("\n替换为消融实验模型（简单拼接融合）...")
    component_paths = {k: str(v) for k, v in COMPONENT_MODELS.items()}
    trainer.model = HierarchicalFloodClassifier_ConcatFusion(
        orientation_model_path=str(ORIENTATION_MODEL),
        component_model_paths=component_paths,
        num_flood_classes=NUM_FLOOD_GRADES,
        freeze_backbone=False,
        use_component_branch=True
    ).to(trainer.device)
    print("✅ 模型替换完成")
    print(f"融合方式: 简单拼接（Concat）")
    print("\n" + "=" * 70)
    print("完全分阶段训练策略（所有参数不冻结）")
    print("=" * 70)
    print("\n解除所有模块的冻结状态...")
    orientation_model = cast(nn.Module, trainer.model.orientation_model)
    for param in orientation_model.parameters():
        param.requires_grad = True
    print("✓ Stage 1 (朝向分类器): 已解除冻结，参数可训练")
    for ori_name, component_model in trainer.model.component_models.items():
        if component_model is not None:
            for param in component_model.parameters():
                param.requires_grad = True
    print("✓ Stage 2 (组件检测器): 已解除冻结，参数可训练")
    for param in trainer.model.flood_classifier.parameters():
        param.requires_grad = True
    print("✓ Stage 3 (洪水等级评估头): 参数可训练")
    trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
    trainer.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"\n参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
    print(f"  冻结参数: 0 (0.00%)")
    print("\n" + "=" * 70)
    print("✅ 使用简单拼接融合模型（消融实验架构）")
    print("📋 配置: 完全分阶段训练（不冻结）| Focal Loss (γ=1.5) | 无类别权重 | 无过采样 | 无数据增强")
    print(f"📊 数据集: 训练{len(trainer.train_dataset)}张 | 验证{len(trainer.val_dataset)}张")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("开始训练（所有参数可训练）")
    print("=" * 70)
    trainer.train()
    print("\n" + "=" * 70)
    print("训练策略消融实验4完成")
    print("=" * 70)

