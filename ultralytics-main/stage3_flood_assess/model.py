"""
分层多任务学习模型

架构：
1. 第一层：车辆朝向识别
2. 第二层：基于朝向的组件检测
3. 第三层：洪水等级分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请安装 ultralytics: pip install ultralytics")

from settings import (
    ORIENTATIONS, NUM_ORIENTATIONS, NUM_FLOOD_GRADES, NUM_COMPONENTS,
    ModelConfig
)
from focal_loss import FocalLoss, AdaptiveFocalLoss


class ConcatFusionModule(nn.Module):
    """简单拼接融合模块"""

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cat([x1, x2], dim=-1)


class FloodClassifierHead(nn.Module):
    """洪水等级分类头 - 融合全局图像特征和组件检测特征"""
    
    def __init__(self, 
                 image_dim: int = 256,
                 component_dim: int = 24,  # 4 components × 6 features
                 num_classes: int = 5,
                 dropout: float = 0.3,
                 fusion_type: str = "gated"):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        if self.fusion_type not in {"gated", "concat"}:
            raise ValueError(
                f"Unsupported fusion_type '{fusion_type}'. "
                "Valid options are 'gated' and 'concat'."
            )
        
        # 图像特征编码器
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # 组件特征编码器
        if self.fusion_type == "gated":
            component_hidden_dim = 64
        else:
            component_hidden_dim = 64  # concat 版本同样保持 64
        self.component_hidden_dim = component_hidden_dim
        self.component_encoder = nn.Sequential(
            nn.Linear(component_dim, component_hidden_dim),
            nn.LayerNorm(component_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(component_hidden_dim, component_hidden_dim),
            nn.LayerNorm(component_hidden_dim),
            nn.GELU(),
        )
        
        # 根据融合方式配置模块
        if self.fusion_type == "gated":
            self.fusion = GatedFusionModule(component_hidden_dim)
            self.image_projection = nn.Linear(128, component_hidden_dim)
            combined_dim = 128 + component_hidden_dim  # 128 + 64 = 192
        else:
            self.fusion = ConcatFusionModule()
            self.image_projection = nn.Linear(128, component_hidden_dim)
            combined_dim = 128 + (component_hidden_dim * 2)  # 128 + 128 = 256
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, image_features: torch.Tensor, 
                component_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, image_dim] 全局图像特征
            component_features: [B, component_dim] 组件检测特征
        Returns:
            logits: [B, num_classes] 洪水等级logits
        """
        # 编码特征
        img_feat = self.image_encoder(image_features)  # [B, 128]
        comp_feat = self.component_encoder(component_features)  # [B, hidden_dim]
        
        if self.fusion_type == "gated":
            assert self.image_projection is not None
            assert self.fusion is not None
            img_feat_proj = self.image_projection(img_feat)  # [B, hidden_dim]
            fused_feat = self.fusion(img_feat_proj, comp_feat)  # [B, hidden_dim]
            combined = torch.cat([img_feat, fused_feat], dim=1)
        else:
            assert self.image_projection is not None
            assert self.fusion is not None
            img_feat_proj = self.image_projection(img_feat)  # [B, hidden_dim]
            concat_feat = self.fusion(img_feat_proj, comp_feat)  # [B, hidden_dim*2]
            combined = torch.cat([img_feat, concat_feat], dim=1)
        
        # 分类
        logits = self.classifier(combined)
        return logits


class HierarchicalFloodModel(nn.Module):
    """
    分层多任务模型主类
    
    工作流程:
    1. 输入图片 -> 朝向识别网络 -> 预测朝向
    2. 根据朝向选择对应的组件检测网络 -> 检测组件位置
    3. 提取全局图像特征 + 组件特征 -> 融合网络 -> 洪水等级分类
    """
    
    def __init__(self,
                 orientation_model_path: str,
                 component_model_paths: Dict[str, str],
                 num_flood_classes: int = NUM_FLOOD_GRADES,
                 freeze_backbone: bool = True,
                 use_component_branch: bool = True,
                 fusion_type: str = "gated"):
        """
        初始化模型
        
        Args:
            orientation_model_path: 朝向分类模型路径
            component_model_paths: 组件检测模型路径字典 {orientation_name: model_path}
            num_flood_classes: 洪水等级类别数
            freeze_backbone: 是否冻结骨干网络
            use_component_branch: 是否使用组件分支
        """
        super().__init__()
        
        self.freeze_backbone = freeze_backbone
        self.use_component_branch = use_component_branch
        self.num_flood_classes = num_flood_classes
        self.fusion_type = fusion_type
        
        # 1. 加载朝向分类模型（YOLO-CLS）
        print(f"[1/6] 正在加载朝向分类模型...")
        print(f"      路径: {orientation_model_path}")
        import time
        start_time = time.time()
        orientation_yolo = YOLO(orientation_model_path)
        # 只保存YOLO的model部分，避免train()方法冲突
        self.orientation_model = orientation_yolo.model
        self.backbone_dim = self._infer_backbone_dim()
        print(f"      ✓ 加载完成 (耗时 {time.time() - start_time:.2f}秒)")
        
        # 冻结朝向模型
        for param in self.orientation_model.parameters():
            param.requires_grad = False
        
        # 2. 加载组件检测模型（YOLO-Detect）
        print(f"\n正在加载5个组件检测模型...")
        self.component_models = {}
        # 保存YOLO对象用于推理，但不作为nn.Module的子模块
        self._component_yolo_models = {}
        for idx, (ori_name, model_path) in enumerate(component_model_paths.items(), start=2):
            if Path(model_path).exists():
                print(f"[{idx}/6] 正在加载 {ori_name} 模型...")
                print(f"      路径: {model_path}")
                start_time = time.time()
                yolo_model = YOLO(model_path)
                # 保存YOLO对象（用于推理）
                self._component_yolo_models[ori_name] = yolo_model
                print(f"      ✓ 加载完成 (耗时 {time.time() - start_time:.2f}秒)")
                # 保存PyTorch模型（用于参数管理）
                self.component_models[ori_name] = yolo_model.model
                # 冻结检测模型
                for param in yolo_model.model.parameters():
                    param.requires_grad = False
            else:
                print(f"  - {ori_name}: 模型不存在，跳过")
                self.component_models[ori_name] = None
                self._component_yolo_models[ori_name] = None
        
        # 3. 洪水等级分类头（可训练）
        component_dim = NUM_COMPONENTS * ModelConfig.component_feature_dim
        if not use_component_branch:
            component_dim = 0
        
        self.flood_classifier = FloodClassifierHead(
            image_dim=self.backbone_dim,
            component_dim=max(component_dim, 1),  # 避免维度为0
            num_classes=num_flood_classes,
            dropout=ModelConfig.dropout_rate,
            fusion_type=fusion_type
        )
        
        print(f"\n✅ 模型构建完成")
        print(f"  - 骨干特征维度: {self.backbone_dim}")
        print(f"  - 组件特征维度: {component_dim}")
        print(f"  - 洪水等级类别数: {num_flood_classes}")
        print(f"  - 使用组件分支: {use_component_branch}")
    
    def _infer_backbone_dim(self) -> int:
        """推断骨干网络特征维度"""
        device = next(self.orientation_model.parameters()).device
        dummy_input = torch.zeros(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features = self._extract_backbone_features(dummy_input)
            pooled = F.adaptive_avg_pool2d(features, (1, 1))
            dim = pooled.flatten(1).shape[1]
        
        return dim
    
    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        
        model = self.orientation_model
        
        # 遍历模型层，提取最后一层特征图
        if hasattr(model, 'model'):
            # 大多数YOLO模型结构
            layers = list(model.model)
            # 去掉最后的分类头
            for layer in layers[:-1]:
                x = layer(x)
            return x
        else:
            raise RuntimeError("无法从YOLO模型提取骨干特征")
    
    @torch.no_grad()
    def _predict_orientation(self, images: torch.Tensor) -> torch.Tensor:
       
        # 使用YOLO模型进行推理
        logits = self.orientation_model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        
        orientation_ids = torch.argmax(logits, dim=1)
        return orientation_ids
    
    @torch.no_grad()
    def _detect_components(self, image: torch.Tensor, orientation_id: int) -> torch.Tensor:
      
        if not self.use_component_branch:
            return torch.zeros(NUM_COMPONENTS * 6, device=image.device)
        
        # 获取对应朝向的YOLO检测模型
        orientation_name = ORIENTATIONS.get(int(orientation_id), 'front')
        detector = self._component_yolo_models.get(orientation_name)
        
        if detector is None:
            return torch.zeros(NUM_COMPONENTS * 6, device=image.device)
        
        # 准备输入（YOLO期望[0,1]范围）
        img_input = image.clamp(0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # 运行检测
        results = detector(img_input, imgsz=640, verbose=False)
        
        if not results or len(results) == 0:
            return torch.zeros(NUM_COMPONENTS * 6, device=image.device)
        
        result = results[0]
        
        # 解析检测结果
        component_feats = torch.zeros(NUM_COMPONENTS, 6, device=image.device)
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy  # [N, 4]
            conf = boxes.conf  # [N]
            cls = boxes.cls.long()  # [N]
            
            # 图片尺寸（用于归一化）
            if hasattr(result, 'orig_shape'):
                H, W = result.orig_shape
            else:
                H, W = 640, 640
            H, W = float(H), float(W)
            
            # 为每个组件类别选择置信度最高的检测框
            for comp_id in range(NUM_COMPONENTS):
                mask = (cls == comp_id)
                if mask.any():
                    # 找到该类别中置信度最高的框
                    comp_boxes = xyxy[mask]
                    comp_confs = conf[mask]
                    max_idx = torch.argmax(comp_confs)
                    
                    box = comp_boxes[max_idx]
                    confidence = comp_confs[max_idx]
                    
                    x1, y1, x2, y2 = box
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    
                    component_feats[comp_id] = torch.tensor([
                        1.0,  # 检测到
                        confidence.item(),
                        cx.item(),
                        cy.item(),
                        w.item(),
                        h.item()
                    ], device=image.device)
        
        return component_feats.flatten()
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
       
        batch_size = images.size(0)
        device = images.device
        
        # 1. 预测朝向
        orientation_ids = self._predict_orientation(images)
        
        # 2. 提取全局图像特征
        backbone_features = self._extract_backbone_features(images)
        image_features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
        image_features = image_features.flatten(1)  # [B, backbone_dim]
        
        # 3. 对每张图片进行组件检测（基于预测的朝向）
        component_features_list = []
        for i in range(batch_size):
            comp_feat = self._detect_components(images[i], orientation_ids[i])
            component_features_list.append(comp_feat)
        
        component_features = torch.stack(component_features_list, dim=0)  # [B, comp_dim]
        
        # 4. 洪水等级分类
        flood_logits = self.flood_classifier(image_features, component_features)
        
        return {
            'flood_grade': flood_logits,
            'orientation_pred': orientation_ids,
        }
    
    def compute_loss(self, 
                    outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    class_weights: Optional[torch.Tensor] = None,
                    label_smoothing: float = 0.0,
                    use_confidence: bool = False,
                    use_focal_loss: bool = False,
                    focal_gamma: float = 2.0) -> Tuple[torch.Tensor, Dict[str, float]]:
       
        # 洪水等级分类损失
        flood_logits = outputs['flood_grade']
        flood_targets = targets['flood_grade']
        
        # 方案1: 使用Focal Loss（最推荐用于类别不平衡）
        if use_focal_loss:
            if use_confidence and 'confidence' in targets:
                # 自适应Focal Loss
                focal_loss_fn = AdaptiveFocalLoss(
                    alpha=class_weights,
                    gamma_high_conf=1.5,
                    gamma_low_conf=3.0
                )
                flood_loss = focal_loss_fn(flood_logits, flood_targets, 
                                          targets['confidence'])
            else:
                # 标准Focal Loss
                focal_loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
                flood_loss = focal_loss_fn(flood_logits, flood_targets)
        
                flood_loss = total_loss / batch_size
        else:
            # 传统方式：固定标签平滑
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            flood_loss = loss_fn(flood_logits, flood_targets)
        
        loss_dict = {
            'flood_loss': flood_loss.item(),
            'total_loss': flood_loss.item(),
        }
        
        return flood_loss, loss_dict


if __name__ == '__main__':
    print("模型定义测试...")
    
    # 测试门控融合模块
    fusion = GatedFusionModule(64)
    x1 = torch.randn(4, 64)
    x2 = torch.randn(4, 64)
    out = fusion(x1, x2)
    print(f"门控融合输出形状: {out.shape}")
    
    # 测试分类头
    classifier = FloodClassifierHead(
        image_dim=256,
        component_dim=24,
        num_classes=5
    )
    img_feat = torch.randn(4, 256)
    comp_feat = torch.randn(4, 24)
    logits = classifier(img_feat, comp_feat)
    print(f"分类头输出形状: {logits.shape}")
    
    print("✅ 模型定义完成")

