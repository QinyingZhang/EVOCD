"""
Helper utility functions
"""
import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Use default font (English)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Filter matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: dict, filepath: str):
    """Save JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {filepath}")


def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_run_directory(base_dir: str) -> str:
    """Create new run directory (train_01, train_02, ...)"""
    os.makedirs(base_dir, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith('train_')]
    if not existing_runs:
        run_id = 1
    else:
        run_ids = []
        for d in existing_runs:
            try:
                run_ids.append(int(d.split('_')[1]))
            except:
                pass
        run_id = max(run_ids) + 1 if run_ids else 1
    
    run_dir = os.path.join(base_dir, f'train_{run_id:02d}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def compute_class_weights(labels: List[int], num_classes: int, 
                          aggressive: bool = True) -> torch.Tensor:
    """
    Compute class weights for class imbalance
    
    Args:
        labels: list of labels
        num_classes: number of classes
        aggressive: whether to use more aggressive weighting
    
    Returns:
        weights tensor
    """
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    total = counts.sum()
    
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    
    if aggressive:
        # More aggressive: use sqrt inverse, amplify minority class weights
        weights = torch.sqrt(total / (num_classes * counts))
        
        # Further boost for very rare classes (<50 samples)
        for i in range(num_classes):
            if counts[i] < 50:
                weights[i] *= 2.0  # Double weight
            elif counts[i] < 100:
                weights[i] *= 1.5  # 1.5x weight
    else:
        # Original: inversely proportional to class frequency
        weights = total / (num_classes * counts)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights


def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """
    Plot training curves
    
    Args:
        history: dict containing training history
        save_path: save path
    """
    # 设置字体为 Arial
    plt.rcParams['font.family'] = 'Arial'
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Flood grade loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_flood_loss'], 'b-', label='Train Flood Loss', linewidth=2)
    ax.plot(epochs, history['val_flood_loss'], 'r-', label='Val Flood Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Flood Classification Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['val_accuracy'], 'g-', label='Val Accuracy', linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Learning rate
    if 'learning_rate' in history:
        ax = axes[1, 1]
        ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('(d) Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved: {save_path}")


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], save_path: str):
    """
    Plot confusion matrix
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        save_path: save path
    """
    # 设置字体为 Arial
    plt.rcParams['font.family'] = 'Arial'
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


def generate_classification_report(y_true: List[int], y_pred: List[int],
                                   class_names: List[str], save_path: str):
    """
    Generate classification report
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        save_path: save path
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   digits=4)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"✅ Classification report saved: {save_path}")
    return report


class AverageMeter:
    """Compute and store average and current value"""
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: number of tolerated epochs
            min_delta: minimum improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']


def format_time(seconds: float) -> str:
    """Format time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count model parameters
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: torch.nn.Module):
    """Print model summary"""
    total, trainable = count_parameters(model)
    print("\n" + "=" * 60)
    print("Model summary")
    print("=" * 60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # 测试
    print("Helper functions test")
    
    # 测试类别权重计算
    labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4]
    weights = compute_class_weights(labels, 5)
    print(f"\nClass weights: {weights}")
    
    # 测试运行目录创建
    test_dir = create_run_directory('./test_runs')
    print(f"Created run directory: {test_dir}")

