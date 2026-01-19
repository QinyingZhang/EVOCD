"""
Dataset class definitions
"""
import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, cast
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from settings import ORIENTATION_TO_ID, FLOOD_GRADES


class FloodDataset(Dataset):
    """
    Flood grade assessment dataset
    
    Data format: CSV file, with columns:
    - image_path: image path
    - orientation: vehicle orientation (name or ID)
    - flood_grade: flood grade (internal index 0-4, display Level 1-5)
    - confidence: annotation confidence (optional, 0-1, default 1.0)
    """
    
    def __init__(self, 
                 csv_path: str,
                 image_root: Optional[str] = None,
                 image_size: int = 640,
                 is_train: bool = True,
                 augment: bool = True,
                 oversample: bool = False,
                 oversample_rates: Optional[Dict[int, int]] = None):
        """
        Args:
            csv_path: CSV file path
            image_root: image root directory (if relative paths in CSV)
            image_size: image size
            is_train: whether this is training set
            augment: whether to use augmentation
            oversample: whether to oversample minority classes
            oversample_rates: repeat factors for each class, e.g. {0: 5, 1: 2}
        """
        self.csv_path = csv_path
        self.image_root = image_root
        self.image_size = image_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.oversample = oversample and is_train
        self.oversample_rates = oversample_rates or {0: 5, 1: 2}
        
        # Load data
        self.samples = self._load_samples()
        
        # Apply oversampling
        if self.oversample:
            original_len = len(self.samples)
            self.samples = self._apply_oversampling()
            print(f"✓ 过采样: {original_len} → {len(self.samples)} 个样本")
        
        # Build transforms
        self.transform = self._build_transform()
        
        print(f"{'Train' if is_train else 'Validation'} set loaded: {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Tuple[str, int, int, float]]:
        """Load sample list (image_path, orientation_id, flood_grade, confidence)"""
        samples = []
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:  # 使用utf-8-sig自动处理BOM
            reader = csv.DictReader(f)
            for row in reader:
                # 图片路径（处理可能的BOM字符）
                img_path = row.get('image_path') or row.get('\ufeffimage_path', '')
                if not img_path:
                    # 尝试获取第一个键（可能是带BOM的image_path）
                    keys = list(row.keys())
                    if keys:
                        img_path = row[keys[0]]
                    else:
                        print(f"Warning: Cannot find image path column, skipping")
                        continue
                if self.image_root and not os.path.isabs(img_path):
                    img_path = os.path.join(self.image_root, img_path)
                
                # 检查文件是否存在
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found, skipping: {img_path}")
                    continue
                
                # Orientation
                orientation = row['orientation']
                if orientation.isdigit():
                    orientation_id = int(orientation)
                else:
                    orientation_id = ORIENTATION_TO_ID.get(orientation)
                    if orientation_id is None:
                        print(f"Warning: Unknown orientation '{orientation}', skipping")
                        continue
                
                # Flood grade
                try:
                    flood_grade = int(row['flood_grade'])
                    if flood_grade < 0 or flood_grade >= len(FLOOD_GRADES):
                        print(f"Warning: Flood grade out of range {flood_grade}, skipping")
                        continue
                except ValueError:
                    print(f"Warning: Invalid flood grade '{row['flood_grade']}', skipping")
                    continue
                
                # Confidence (optional)
                confidence = 1.0  # default value
                if 'confidence' in row:
                    try:
                        confidence = float(row['confidence'])
                        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
                    except ValueError:
                        print(f"Warning: Invalid confidence '{row['confidence']}', using default value 1.0")
                
                samples.append((img_path, orientation_id, flood_grade, confidence))
        
        if len(samples) == 0:
            raise ValueError(f"No valid samples found: {self.csv_path}")
        
        return samples
    
    def _apply_oversampling(self) -> List[Tuple[str, int, int, float]]:
        """Apply oversampling to minority class samples"""
        from collections import Counter
        
        # Count samples per class
        flood_grades = [s[2] for s in self.samples]
        grade_counts = Counter(flood_grades)
        
        print("\nOriginal distribution:")
        for grade in sorted(grade_counts.keys()):
            print(f"  Level {grade+1} (index {grade}): {grade_counts[grade]} samples")
        
        # Apply oversampling
        oversampled = []
        for sample in self.samples:
            flood_grade = sample[2]
            repeat_times = self.oversample_rates.get(flood_grade, 1)
            
            # Repeat samples
            for _ in range(repeat_times):
                oversampled.append(sample)
        
        # Count distribution after oversampling
        new_grades = [s[2] for s in oversampled]
        new_counts = Counter(new_grades)
        
        print("\nDistribution after oversampling:")
        for grade in sorted(new_counts.keys()):
            print(f"  Level {grade+1} (index {grade}): {new_counts[grade]} samples (×{self.oversample_rates.get(grade, 1)})")
        print()
        
        return oversampled
    
    def _build_transform(self):
        """Build image transforms"""
        if self.augment:
            # Training set: with augmentation
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Validation/test set: no augmentation
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Returns:
            dict with keys:
            - 'image': [3, H, W] tensor
            - 'orientation': scalar tensor
            - 'flood_grade': scalar tensor
            - 'confidence': scalar tensor
            - 'image_path': str
        """
        img_path, orientation_id, flood_grade, confidence = self.samples[idx]
        
        # Load image
        try:
            pil_image = Image.open(img_path).convert('RGB')
            image = cast(torch.Tensor, self.transform(pil_image))
        except Exception as e:
            print(f"Error: Failed to load image {img_path}: {e}")
            # Return all-zero image as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'image': image,
            'orientation': torch.tensor(orientation_id, dtype=torch.long),
            'flood_grade': torch.tensor(flood_grade, dtype=torch.long),
            'confidence': torch.tensor(confidence, dtype=torch.float32),
            'image_path': img_path,
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        from collections import Counter
        import numpy as np
        
        orientations = [s[1] for s in self.samples]
        flood_grades = [s[2] for s in self.samples]
        confidences = [s[3] for s in self.samples]
        
        return {
            'total_samples': len(self.samples),
            'orientation_distribution': dict(Counter(orientations)),
            'flood_grade_distribution': dict(Counter(flood_grades)),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
            },
        }


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True) -> torch.utils.data.DataLoader:
    """Create data loader"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


if __name__ == '__main__':
    # Test dataset class
    print("Testing dataset class...")
    
    # Create test CSV
    test_csv = 'test_dataset.csv'
    with open(test_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'orientation', 'flood_grade'])
        # Actual image paths needed for testing here
    
    print("Dataset class definition complete")

