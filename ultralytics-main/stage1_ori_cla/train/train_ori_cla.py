from ultralytics import YOLO
import os
from pathlib import Path

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型（升级到 yolo11s-cls 提升容量）
model = YOLO('yolo11n-cls.pt')

# Use relative paths for training
if __name__ == '__main__':
    dataset_path = Path(current_dir) / 'dataset'
    save_dir = Path(current_dir) / 'runs'  # 指定保存路径
    
    print(f"Current directory: {current_dir}")
    print(f"Dataset path: {dataset_path}")
    print(f"Save directory: {save_dir}")
    print(f"Train directory exists: {(dataset_path / 'train').exists()}")
    print(f"Val directory exists: {(dataset_path / 'val').exists()}")
    
    # Check classes and sample count in train directory (count common extensions)
    train_dir = dataset_path / 'train'
    if train_dir.exists():
        print("Training classes found:")
        exts = ('*.jpg', '*.jpeg', '*.png', '*.webp')
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                count = 0
                for ext in exts:
                    count += len(list(class_dir.glob(ext)))
                print(f"- {class_dir.name}: {count} images")
    
    # Recommended settings: larger model and resolution, moderate batch, longer patience, mild regularization, cosine annealing
    results = model.train(
        data=str(dataset_path),
        epochs=100,
        imgsz=256,
        batch=32,            # 显存不足可降到 24/16
        workers=4,
        device=0,            # 使用GPU:0
        project=str(save_dir),
        name='train_01',
        exist_ok=True,
        save=True,
        save_period=10,
        patience=25,
        lr0=0.0015,
        weight_decay=0.001,
        cos_lr=True
    )