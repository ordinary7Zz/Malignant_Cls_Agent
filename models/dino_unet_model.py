"""
DINO-UNet 多任务模型包装
基于 DINOv3 + UNet 的多任务模型，用于甲状腺结节分类
"""

import sys
import os
import torch
import torch.fx
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from models.base_model import BaseClassificationModel, ModelOutput
from model_architectures import DINOv3_S_UNet_MULTITASK


class DINOUNetModel(BaseClassificationModel):
    """
    DINO-UNet 多任务模型
    - 输入：仅需要图像（RGB）
    - 输出：二分类(0/1) + TI-RADS 分类
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_tirads: bool = False  # 是否使用 TI-RADS 分类而非二分类(0/1)
    ):
        """
        初始化 DINO-UNet 模型
        
        Args:
            model_path: 模型权重路径
            device: 运行设备
            use_tirads: True=使用 TI-RADS 5分类, False=使用二分类(0/1)
        """
        super().__init__(
            model_name="DINO_UNet_MultiTask",
            model_path=model_path,
            requires_mask=False
        )
        
        self.device = device
        self.use_tirads = use_tirads
        
        # 类别名称
        if use_tirads:
            self.class_names = ["TI-RADS 1", "TI-RADS 2", "TI-RADS 3", 
                                "TI-RADS 4", "TI-RADS 5"]
        else:
            self.class_names = ["0", "1"]
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """加载 DINO-UNet 模型"""
        print(f"加载 {self.model_name} 从 {self.model_path}")
        
        try:
            # 先检查权重文件，判断是否需要使用dilation
            use_dilation = False
            state_dict = None
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 处理不同的checkpoint格式
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 检查state_dict中是否包含dilation相关的键
                dilation_keys = [key for key in state_dict.keys() if 'dilate' in key]
                if len(dilation_keys) > 0:
                    use_dilation = True
                    print(f"    检测到dilation参数，使用 use_dilation=True")
            
            # 根据权重文件创建模型
            self.model = DINOv3_S_UNet_MULTITASK(pretrained=False, use_dilation=use_dilation)
            
            # 加载权重
            if state_dict is not None:
                # 尝试加载权重，如果strict模式失败，则使用非strict模式
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    print(f"✓ 成功加载权重")
                except RuntimeError as e:
                    # 如果strict模式失败，尝试非strict模式
                    if "Unexpected key(s)" in str(e) or "Missing key(s)" in str(e):
                        print(f"    警告: 权重键不完全匹配，使用非严格模式加载")
                        # 获取模型当前state_dict的键
                        model_keys = set(self.model.state_dict().keys())
                        state_dict_keys = set(state_dict.keys())
                        
                        # 只加载匹配的键
                        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                        missing_keys = model_keys - state_dict_keys
                        unexpected_keys = state_dict_keys - model_keys
                        
                        if missing_keys:
                            print(f"    缺失的键: {len(missing_keys)} 个（将使用随机初始化）")
                        if unexpected_keys:
                            print(f"    额外的键: {len(unexpected_keys)} 个（将被忽略）")
                        
                        self.model.load_state_dict(filtered_state_dict, strict=False)
                        print(f"✓ 成功加载权重（非严格模式）")
                    else:
                        raise
            else:
                print(f"⚠️  警告: 权重文件不存在 {self.model_path}，使用随机初始化")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            raise
    
    def preprocess(self, image: np.ndarray, mask=None):
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C) numpy array, RGB格式
            mask: 不需要掩码
            
        Returns:
            处理后的张量
        """
        # 转换为 PIL Image
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # 应用变换
        tensor = self.transform(pil_image)
        
        # 添加批次维度
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray, mask=None):
        """
        运行推理
        
        Args:
            image: 输入图像 (H, W, C)
            mask: 不需要掩码
            
        Returns:
            ModelOutput 包含预测结果
        """
        self.validate_inputs(image, mask)
        
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            seg_out, benign_malignant, tirads = self.model(input_tensor)
        
        # 处理输出
        if self.use_tirads:
            # 使用 TI-RADS 5分类
            probs = torch.softmax(tirads, dim=1)[0].cpu().numpy()
            predictions = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probs)
            }
        else:
            # 使用二分类(0/1)
            prob_1 = torch.sigmoid(benign_malignant)[0, 0].cpu().item()
            prob_0 = 1.0 - prob_1
            predictions = {
                "0": float(prob_0),
                "1": float(prob_1)
            }
        
        # 找到最高置信度的类别
        top_class = max(predictions.items(), key=lambda x: x[1])
        
        metadata = {
            "framework": "pytorch",
            "backbone": "DINOv3 + UNet",
            "task": "TI-RADS" if self.use_tirads else "binary_classification",
            "device": self.device
        }

        if not self.use_tirads:
            metadata["positive_class_index"] = 1
        
        # 添加训练数据设备信息（如果存在）
        if hasattr(self, 'training_data_devices') and self.training_data_devices:
            metadata['training_data_devices'] = self.training_data_devices
        
        # 添加验证集性能指标（如果存在）
        if hasattr(self, 'validation_metrics') and self.validation_metrics:
            metadata['validation_metrics'] = self.validation_metrics
        
        # 添加多测试集性能（兼容旧属性 base_dataset_performance）
        if hasattr(self, 'test_set_performance') and self.test_set_performance:
            metadata['test_set_performance'] = self.test_set_performance
        elif hasattr(self, 'base_dataset_performance') and self.base_dataset_performance:
            metadata['test_set_performance'] = self.base_dataset_performance
        
        # 添加数据集规模信息（如果存在）
        if hasattr(self, 'dataset_info') and self.dataset_info:
            metadata['dataset_info'] = self.dataset_info
        
        return ModelOutput(
            model_name=self.model_name,
            predictions=predictions,
            top_class=top_class[0],
            top_confidence=top_class[1],
            requires_mask=self.requires_mask,
            metadata=metadata
        )

