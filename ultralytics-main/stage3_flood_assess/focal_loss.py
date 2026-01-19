"""
Focal Loss - 专门用于处理类别不平衡的损失函数
Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    其中：
    - p_t 是模型预测正确类别的概率
    - α_t 是类别权重
    - γ 是focusing参数（通常取2.0）
    
    当样本容易分类时（p_t接近1），(1-p_t)^γ 接近0，损失被down-weight
    当样本难分类时（p_t接近0），(1-p_t)^γ 接近1，损失保持高权重
    
    这样模型会更关注难样本（通常是少数类）
    """
    
    def __init__(self, 
                 alpha: torch.Tensor = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: class weights [num_classes], optional
            gamma: focusing parameter, larger values focus more on hard samples
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, num_classes] model output logits
            targets: [B] target class
        
        Returns:
            loss: scalar
        """
        # Calculate cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate predicted probability
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B] probability of the true class
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss - combines confidence and Focal Loss

    For low-confidence samples, use larger gamma (focus more on hard samples)
    For high-confidence samples, use smaller gamma
    """
    
    def __init__(self,
                 alpha: torch.Tensor = None,
                 gamma_high_conf: float = 1.5,
                 gamma_low_conf: float = 3.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: class weights
            gamma_high_conf: gamma for high-confidence samples
            gamma_low_conf: gamma for low-confidence samples
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma_high = gamma_high_conf
        self.gamma_low = gamma_low_conf
        self.reduction = reduction
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor,
                confidences: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, num_classes]
            targets: [B]
            confidences: [B] annotation confidence (0-1)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Adjust gamma based on confidence
        if confidences is not None:
            # Confidence is low → gamma is large (more focus)
            # Confidence is high → gamma is small (less focus)
            gamma = self.gamma_low + (self.gamma_high - self.gamma_low) * confidences
        else:
            gamma = torch.full_like(p_t, self.gamma_high)
        
        # Focal weight with adaptive gamma
        focal_weight = (1 - p_t) ** gamma
        loss = focal_weight * ce_loss
        
        # Class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == '__main__':
    # 测试Focal Loss
    print("测试Focal Loss...")
    
    # 模拟数据
    batch_size = 8
    num_classes = 5
    
    # 模拟类别不平衡的权重
    alpha = torch.tensor([10.0, 5.0, 3.0, 2.0, 1.0])
    
    # 创建损失函数
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss(weight=alpha)
    
    # 随机输入
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失
    fl = focal_loss(inputs, targets)
    cel = ce_loss(inputs, targets)
    
    print(f"Focal Loss: {fl.item():.4f}")
    print(f"CE Loss:    {cel.item():.4f}")
    print(f"✅ Focal Loss测试完成")
    
    # 测试自适应Focal Loss
    print("\n测试Adaptive Focal Loss...")
    confidences = torch.rand(batch_size)  # 随机置信度
    adaptive_fl = AdaptiveFocalLoss(alpha=alpha)
    afl = adaptive_fl(inputs, targets, confidences)
    print(f"Adaptive Focal Loss: {afl.item():.4f}")
    print(f"✅ Adaptive Focal Loss测试完成")
