"""
Inference script - Use trained model for flood grade prediction
"""
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from settings import (
    ORIENTATIONS,
    FLOOD_GRADES_CN,
    FLOOD_GRADES,
    COMPONENT_MODELS,
    ORIENTATION_MODEL,
)
from model import HierarchicalFloodModel
from helpers import format_time
import time


class FloodGradePredictor:
    """洪水等级预测器"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda',
                 image_size: int = 640):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 'cuda' or 'cpu'
            image_size: 图片尺寸
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # 加载模型
        print(f"加载模型: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # 构建图像变换
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 预测器初始化完成 (设备: {self.device})")
    
    def _load_model(self, checkpoint_path: str) -> HierarchicalFloodModel:
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        fusion_type = self._infer_fusion_type(state_dict)
        
        # 构建模型
        component_models_dict = {k: str(v) for k, v in COMPONENT_MODELS.items()}
        model = HierarchicalFloodModel(
            orientation_model_path=str(ORIENTATION_MODEL),
            component_model_paths=component_models_dict,
            num_flood_classes=5,
            freeze_backbone=True,
            use_component_branch=True,
            fusion_type=fusion_type,
        ).to(self.device)
        
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _infer_fusion_type(state_dict: Dict[str, torch.Tensor]) -> str:
        """根据checkpoint推断融合方式"""
        for key in state_dict.keys():
            if key.startswith("flood_classifier.fusion."):
                return "gated"
        return "concat"
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图片"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # [1, 3, H, W]

    def _detect_component_boxes(
        self,
        image_path: str,
        orientation_id: int,
    ) -> Optional[List[List[float]]]:
        """
        获取组件检测框（用于可视化）

        Returns:
            List[[x1, y1, x2, y2, conf, cls_id]] or None
        """
        orientation_name = ORIENTATIONS.get(int(orientation_id), 'front')
        detector = self.model._component_yolo_models.get(orientation_name)

        if detector is None:
            return None

        # 运行检测
        device_str = str(self.device)
        results = detector(image_path, imgsz=640, verbose=False, device=device_str)

        if not results:
            return None

        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        component_boxes = []
        for (x1, y1, x2, y2), c, cls_id in zip(xyxy, conf, cls):
            component_boxes.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(c),
                int(cls_id),
            ])
        return component_boxes
    
    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        预测单张图片的洪水等级
        
        Args:
            image_path: 图片路径
        Returns:
            dict with keys:
            - 'flood_grade': 预测的洪水等级 (1-5，显示值)
            - 'flood_grade_idx': 内部索引 (0-4，用于访问数组)
            - 'flood_name': 洪水等级中文名称
            - 'confidence': 置信度
            - 'probabilities': 所有类别的概率 [5]
            - 'orientation': 预测的车辆朝向
            - 'orientation_name': 车辆朝向名称
        """
        # 预处理
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # 推理
        outputs = self.model(image_tensor)
        
        # 解析结果
        flood_logits = outputs['flood_grade'][0]  # [5]
        probabilities = torch.softmax(flood_logits, dim=0)
        
        flood_grade_idx = torch.argmax(probabilities).item()  # 内部索引 0-4
        flood_grade = flood_grade_idx + 1  # 显示值 1-5
        confidence = probabilities[flood_grade_idx].item()
        
        orientation_id = outputs['orientation_pred'][0].item()
        orientation_name = ORIENTATIONS.get(orientation_id, 'unknown')

        component_boxes = self._detect_component_boxes(image_path, orientation_id)
        
        return {
            'flood_grade': flood_grade,  # 显示值 1-5
            'flood_grade_idx': flood_grade_idx,  # 内部索引 0-4
            'flood_name': FLOOD_GRADES_CN[flood_grade_idx],
            'flood_name_en': FLOOD_GRADES[flood_grade_idx],
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'orientation': orientation_id,
            'orientation_name': orientation_name,
            'component_boxes': component_boxes,
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                result['image_path'] = img_path
                result['success'] = True
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'success': False,
                    'error': str(e)
                })
        return results
    
    def visualize_prediction(self, image_path: str, output_path: str = None):
        """
        可视化预测结果
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径（如果为None，则显示）
        """
        # 预测
        result = self.predict(image_path)
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 创建画布
        canvas_h = h + 200
        canvas = np.ones((canvas_h, w, 3), dtype=np.uint8) * 255
        canvas[:h] = image
        
        # 绘制文本区域
        info_y = h + 30
        
        # 标题
        title = f"洪水等级评估结果"
        cv2.putText(canvas, title, (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 洪水等级
        info_y += 40
        flood_text = f"等级: Level {result['flood_grade']} - {result['flood_name']}"
        cv2.putText(canvas, flood_text, (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 置信度
        info_y += 35
        conf_text = f"置信度: {result['confidence']:.2%}"
        cv2.putText(canvas, conf_text, (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        
        # 车辆朝向
        info_y += 35
        ori_text = f"车辆朝向: {result['orientation_name']}"
        cv2.putText(canvas, ori_text, (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        
        # 概率分布（右侧）
        prob_x = w // 2 + 50
        prob_y = h + 30
        cv2.putText(canvas, "各等级概率:", (prob_x, prob_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        for i, prob in enumerate(result['probabilities']):
            prob_y += 30
            prob_text = f"Level {i+1}-{FLOOD_GRADES_CN[i][:4]}: {prob:.1%}"
            color = (0, 0, 255) if i == result['flood_grade_idx'] else (128, 128, 128)
            cv2.putText(canvas, prob_text, (prob_x, prob_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 转换颜色空间
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        # 保存或显示
        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"✅ 结果已保存: {output_path}")
        else:
            cv2.imshow("Flood Grade Prediction", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='洪水等级预测')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图片路径')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图片目录（批量预测）')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = FloodGradePredictor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 单张图片
    if args.image:
        print(f"\n预测图片: {args.image}")
        result = predictor.predict(args.image)
        
        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(f"洪水等级: Level {result['flood_grade']} - {result['flood_name']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"车辆朝向: {result['orientation_name']}")
        print("\n各等级概率:")
        for i, prob in enumerate(result['probabilities']):
            print(f"  Level {i+1} ({FLOOD_GRADES_CN[i]}): {prob:.2%}")
        
        # 可视化
        if args.visualize:
            output_path = os.path.join(args.output_dir, 
                                      f"pred_{Path(args.image).name}")
            predictor.visualize_prediction(args.image, output_path)
    
    # 批量预测
    elif args.image_dir:
        print(f"\n批量预测目录: {args.image_dir}")
        
        # 收集图片
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [str(f) for f in Path(args.image_dir).iterdir()
                      if f.suffix.lower() in image_exts]
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 预测
        start_time = time.time()
        results = predictor.predict_batch(image_paths)
        elapsed = time.time() - start_time
        
        # 统计
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"\n完成预测: {success_count}/{len(results)} 成功")
        print(f"用时: {format_time(elapsed)}")
        print(f"平均速度: {len(results)/elapsed:.2f} 张/秒")
        
        # 保存结果
        if args.output_dir:
            import csv
            csv_path = os.path.join(args.output_dir, 'predictions.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'flood_grade', 'flood_name', 
                               'confidence', 'orientation'])
                
                for r in results:
                    if r.get('success'):
                        writer.writerow([
                            r['image_path'],
                            r['flood_grade'],
                            r['flood_name'],
                            f"{r['confidence']:.4f}",
                            r['orientation_name']
                        ])
            
            print(f"✅ 结果已保存: {csv_path}")
            
            # 可视化部分结果
            if args.visualize:
                print("\n可视化前10个结果...")
                for i, r in enumerate(results[:10]):
                    if r.get('success'):
                        output_path = os.path.join(args.output_dir,
                                                  f"pred_{i:03d}_{Path(r['image_path']).name}")
                        predictor.visualize_prediction(r['image_path'], output_path)
    
    else:
        print("请指定 --image 或 --image_dir")


if __name__ == '__main__':
    main()

