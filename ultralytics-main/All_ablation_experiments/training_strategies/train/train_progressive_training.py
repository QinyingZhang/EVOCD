"""
训练策略消融实验3：完全分阶段训练策略（不冻结）
- 策略：分阶段训练但不冻结，所有参数都可训练
- 损失函数：策略三（Focal Loss + 类别权重，无过采样，无数据增强）
- 模型架构：简单拼接融合
"""
import argparse
import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# 0: training_strategies, 1: All_ablation_experiments, 2: ultralytics-main
PROJECT_ROOT = SCRIPT_DIR.parents[2]
HIER_MULT_DIR = PROJECT_ROOT / 'hier_mult'
if str(HIER_MULT_DIR) not in sys.path:
    sys.path.insert(0, str(HIER_MULT_DIR))
# 确保包导入可用（hier_mult.*）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 确保能导入 hier_mult/archived 下的模型
HIER_MULT_ARCHIVED = HIER_MULT_DIR / 'archived'
if str(HIER_MULT_ARCHIVED) not in sys.path:
    sys.path.insert(0, str(HIER_MULT_ARCHIVED))

# 使用包路径导入，避免相对路径问题
import importlib.util

# 动态加载 archived/model_concat_fusion.py
spec_model = importlib.util.spec_from_file_location(
    "model_concat_fusion", str(HIER_MULT_DIR / "archived" / "model_concat_fusion.py")
)
assert spec_model is not None and spec_model.loader is not None
model_module = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(model_module)
HierarchicalFloodClassifier_ConcatFusion = model_module.HierarchicalFloodClassifier_ConcatFusion

# 直接从 hier_mult 目录加载 settings.py
spec_settings = importlib.util.spec_from_file_location(
    "settings", str(HIER_MULT_DIR / "settings.py")
)
assert spec_settings is not None and spec_settings.loader is not None
settings = importlib.util.module_from_spec(spec_settings)
spec_settings.loader.exec_module(settings)
ORIENTATION_MODEL = settings.ORIENTATION_MODEL
COMPONENT_MODELS = settings.COMPONENT_MODELS
NUM_FLOOD_GRADES = settings.NUM_FLOOD_GRADES

# Trainer 从 hier_mult/train.py
spec_train = importlib.util.spec_from_file_location(
    "train_module", str(HIER_MULT_DIR / "train.py")
)
assert spec_train is not None and spec_train.loader is not None
train_module = importlib.util.module_from_spec(spec_train)
spec_train.loader.exec_module(train_module)
Trainer = train_module.Trainer


if __name__ == '__main__':
    print("=" * 70)
    print("训练策略消融实验3：完全分阶段训练策略（不冻结）")
    print("策略: 分阶段训练，但所有参数都可训练（不冻结）")
    print("=" * 70)
    
    # 获取数据集目录
    DATA_DIR = PROJECT_ROOT / 'All_ablation_experiments' / 'data'
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练策略消融：完全分阶段训练（不冻结）')
    
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
    
    # 模型参数 - 关键：不冻结任何模块
    parser.add_argument('--freeze_backbone', type=bool, default=False)  # 不冻结骨干网络
    parser.add_argument('--use_component_branch', type=bool, default=True)
    
    # 数据增强 - 策略三：无数据增强，无过采样
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--oversample', type=bool, default=False)
    parser.add_argument('--oversample_level0', type=int, default=1)
    parser.add_argument('--oversample_level1', type=int, default=1)
    
    # 损失函数 - 策略三：Focal Loss + 类别权重
    parser.add_argument('--use_class_weights', type=bool, default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--use_focal_loss', type=bool, default=True)
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
    parser.add_argument('--runs_dir', type=str, default=str(SCRIPT_DIR / 'runs' / 'progressive_training'))
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 替换模型为简单拼接融合版本
    print("\n替换为消融实验模型（简单拼接融合）...")
    component_paths = {k: str(v) for k, v in COMPONENT_MODELS.items()}
    
    trainer.model = HierarchicalFloodClassifier_ConcatFusion(
        orientation_model_path=str(ORIENTATION_MODEL),
        component_model_paths=component_paths,
        num_flood_classes=NUM_FLOOD_GRADES,
        freeze_backbone=False,  # 不冻结，使用完全分阶段训练
        use_component_branch=True
    ).to(trainer.device)
    
    print("✅ 模型替换完成")
    print(f"融合方式: 简单拼接（Concat）")
    
    # 完全分阶段训练策略：解除所有模块的冻结
    print("\n" + "=" * 70)
    print("完全分阶段训练策略（所有参数不冻结）")
    print("=" * 70)
    print("\n解除所有模块的冻结状态...")
    
    # 解除朝向模型的冻结
    for param in trainer.model.orientation_model.parameters():  # type: ignore
        param.requires_grad = True
    print("✓ Stage 1 (朝向分类器): 已解除冻结，参数可训练")
    
    # 解除组件检测模型的冻结
    for ori_name, component_model in trainer.model.component_models.items():
        if component_model is not None:
            for param in component_model.parameters():
                param.requires_grad = True
    print("✓ Stage 2 (组件检测器): 已解除冻结，参数可训练")
    
    # 分类头始终可训练
    for param in trainer.model.flood_classifier.parameters():
        param.requires_grad = True
    print("✓ Stage 3 (洪水等级评估头): 参数可训练")
    
    # 重新创建优化器（包含所有参数）
    trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
    trainer.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    
    print(f"\n参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
    print(f"  冻结参数: 0 (0.00%)")
    
    print("\n" + "=" * 70)
    print("分阶段训练说明:")
    print("=" * 70)
    print("策略: 分阶段训练但不冻结")
    print("• Stage 1 和 Stage 2 使用预训练模型初始化")
    print("• 训练 Stage 3 时，所有参数都可训练（不冻结）")
    print("• 这允许模型在端到端学习中自适应调整所有模块")
    
    print("\n" + "=" * 70)
    print("✅ 使用简单拼接融合模型（消融实验架构）")
    print("📋 配置: 完全分阶段训练（不冻结） | Focal Loss (γ=1.5) | 类别权重 | 无数据增强 | 无过采样")
    print(f"📊 数据集: 训练{len(trainer.train_dataset)}张 | 验证{len(trainer.val_dataset)}张")
    print("=" * 70)
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练（所有参数可训练）")
    print("=" * 70)
    trainer.train()
    
    print("\n" + "=" * 70)
    print("训练策略消融实验3（完全分阶段训练）完成")
    print("=" * 70)
