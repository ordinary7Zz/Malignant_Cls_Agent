"""
AutoGluon + PyRadiomics 模型包装
基于 PyRadiomics 特征提取和 AutoGluon 表格学习的分类模型
"""

import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import tempfile
from typing import Optional
import pathlib
import pickle

from models.base_model import BaseClassificationModel, ModelOutput


# 全局修复 PosixPath 问题（Windows 系统加载 Linux 训练的模型）
if sys.platform == 'win32':
    # 保存原始的 PosixPath（如果存在）
    _original_posix_path = getattr(pathlib, 'PosixPath', None)
    
    # 创建一个兼容的 PosixPath 类
    class PosixPath(pathlib.WindowsPath):
        """Windows 上的 PosixPath 兼容类"""
        pass
    
    # 替换 pathlib.PosixPath
    pathlib.PosixPath = PosixPath
    
    # 创建自定义的 Unpickler 类来处理 PosixPath
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            """修补的 find_class 方法，处理 PosixPath"""
            if module == 'pathlib' and name == 'PosixPath':
                return PosixPath
            return super().find_class(module, name)
    
    # 保存原始的 pickle.load
    _original_pickle_load = pickle.load
    
    def custom_pickle_load(file, **kwargs):
        """使用自定义 Unpickler 的 pickle.load"""
        return CustomUnpickler(file, **kwargs).load()
    
    # 替换 pickle.load
    pickle.load = custom_pickle_load


class AutoGluonRadiomicsModel(BaseClassificationModel):
    """
    AutoGluon + PyRadiomics 模型
    - 输入：需要图像和掩码
    - 输出：良恶性二分类
    """
    
    def __init__(
        self,
        model_dir: str,
        radiomics_config: str = None
    ):
        """
        初始化 AutoGluon + PyRadiomics 模型
        
        Args:
            model_dir: AutoGluon 模型目录（包含 predictor.pkl）
            radiomics_config: PyRadiomics 配置文件路径
        """
        super().__init__(
            model_name="AutoGluon_PyRadiomics",
            model_path=model_dir,
            requires_mask=True  # 需要掩码
        )
        
        self.model_dir = model_dir
        
        # 设置默认的 radiomics 配置路径
        if radiomics_config is None:
            radiomics_config = str(
                Path(__file__).resolve().parents[1]
                / "config"
                / "radiomics_2d.yaml"
            )
        self.radiomics_config = radiomics_config
        
        # 类别名称
        self.class_names = ["0", "1"]
        
        self.predictor = None
        self.extractor = None
    
    def load_model(self):
        """加载 AutoGluon 模型和 PyRadiomics 提取器"""
        print(f"加载 {self.model_name} 从 {self.model_dir}")
        
        try:
            # 加载 AutoGluon predictor
            # PosixPath 修复已在模块级别应用
            from autogluon.tabular import TabularPredictor
            
            if os.path.exists(self.model_dir):
                self.predictor = TabularPredictor.load(self.model_dir)
                print(f"✓ AutoGluon 模型加载成功")
            else:
                raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
            
            # 初始化 PyRadiomics 提取器
            from radiomics import featureextractor
            import SimpleITK as sitk
            
            if os.path.exists(self.radiomics_config):
                self.extractor = featureextractor.RadiomicsFeatureExtractor(
                    self.radiomics_config
                )
                print(f"✓ PyRadiomics 提取器初始化成功")
            else:
                print(f"⚠️  配置文件不存在: {self.radiomics_config}")
                self.extractor = featureextractor.RadiomicsFeatureExtractor()
                print(f"✓ 使用默认 PyRadiomics 配置")
                
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            raise
    
    def _extract_radiomics_features(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        spacing: tuple = (1.0, 1.0)
    ) -> dict:
        """
        从图像和掩码中提取 PyRadiomics 特征
        
        Args:
            image: 灰度图像 (H, W)
            mask: 二值掩码 (H, W)
            spacing: 像素间距
            
        Returns:
            特征字典
        """
        import SimpleITK as sitk
        
        # 确保是灰度图像
        if image.ndim == 3:
            # 转换为灰度
            image = np.mean(image, axis=2).astype(np.uint8)
        
        # 确保掩码是二值的
        mask_binary = (mask > 0).astype(np.uint8)
        
        # 检查掩码是否为空
        if mask_binary.sum() == 0:
            raise ValueError("掩码为空，无法提取特征")
        
        # 转换为 SimpleITK 格式
        image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask_binary.astype(np.uint8))
        
        image_sitk.SetSpacing(spacing)
        mask_sitk.SetSpacing(spacing)
        
        # 提取特征
        try:
            features = self.extractor.execute(image_sitk, mask_sitk, label=1)
            
            # 过滤出数值特征
            numeric_features = {}
            for key, value in features.items():
                # 排除诊断信息
                if not key.startswith('diagnostics_'):
                    try:
                        numeric_features[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            return numeric_features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            raise
    
    def preprocess(self, image: np.ndarray, mask: np.ndarray):
        """
        预处理：提取 radiomics 特征（单张图像）
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            mask: 分割掩码 (H, W)
            
        Returns:
            特征DataFrame
        """
        # 提取特征
        features = self._extract_radiomics_features(image, mask)
        
        # 转换为 DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
    
    def preprocess_batch(self, images: list, masks: list, show_progress: bool = True):
        """
        批量预处理：一次性提取多张图像的 radiomics 特征
        
        注意：PyRadiomics 本身不支持批量处理，此方法内部仍逐张提取特征，
        但统一管理进度和错误处理，最后批量转换为 DataFrame。
        
        Args:
            images: 图像列表 [(H, W) 或 (H, W, C), ...]
            masks: 掩码列表 [(H, W), ...]
            show_progress: 是否显示进度
            
        Returns:
            特征DataFrame (n_samples, n_features)
        """
        if len(images) != len(masks):
            raise ValueError("图像和掩码数量必须匹配")
        
        all_features = []
        failed_count = 0
        
        # PyRadiomics 不支持真正的批量处理，只能逐张提取
        # 但这里统一管理，最后批量转换为 DataFrame
        for idx, (image, mask) in enumerate(zip(images, masks)):
            if show_progress and (idx + 1) % 50 == 0 or idx == 0 or idx == len(images) - 1:
                print(f"  提取特征进度: [{idx+1}/{len(images)}]...", end="\r")
            
            try:
                features = self._extract_radiomics_features(image, mask)
                all_features.append(features)
            except Exception as e:
                if show_progress:
                    print(f"\n  ⚠️  图像 {idx+1} 特征提取失败: {e}")
                # 使用空特征或默认值
                all_features.append({})
                failed_count += 1
        
        if show_progress:
            print(f"  特征提取完成: {len(images)} 张图像 ({failed_count} 张失败)    ")
        
        # 转换为 DataFrame
        features_df = pd.DataFrame(all_features)
        
        # 填充缺失值（如果有失败的样本）
        if features_df.isnull().any().any():
            features_df = features_df.fillna(0)
        
        return features_df
    
    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        运行推理（单张图像）
        
        Args:
            image: 输入图像 (H, W, C)
            mask: 分割掩码 (H, W) - 必需
            
        Returns:
            ModelOutput 包含预测结果
        """
        self.validate_inputs(image, mask)
        
        # 提取特征
        try:
            features_df = self.preprocess(image, mask)
        except Exception as e:
            print(f"特征提取失败: {e}")
            # 返回一个不确定的结果
            return ModelOutput(
                model_name=self.model_name,
                predictions={"0": 0.5, "1": 0.5},
                top_class="0",
                top_confidence=0.5,
                requires_mask=self.requires_mask,
                metadata={"error": str(e)}
            )
        
        # 预测
        try:
            # 获取预测概率
            proba = self.predictor.predict_proba(features_df)
            
            # 提取概率值
            if hasattr(proba, 'values'):
                proba_array = proba.values[0]
            else:
                proba_array = proba[0]
            
            # 构建预测字典
            if len(proba_array) == 2:
                prob_class_0 = float(proba_array[0])
                prob_class_1 = float(proba_array[1])
            else:
                # 如果只有一个值，按 class_1 概率解释
                prob_class_1 = float(proba_array[0])
                prob_class_0 = 1.0 - prob_class_1
            
            predictions = {
                "0": prob_class_0,
                "1": prob_class_1
            }
            
            # 找到最高置信度的类别
            top_class = max(predictions.items(), key=lambda x: x[1])
            
            # 获取最佳模型名称（如果可用）
            try:
                best_model = self.predictor.model_best
            except:
                best_model = "unknown"
            
            metadata = {
                "framework": "AutoGluon + PyRadiomics",
                "n_features": len(features_df.columns),
                "best_model": best_model
            }
            
            # 添加训练数据设备信息（如果存在）
            if hasattr(self, 'training_data_devices') and self.training_data_devices:
                metadata['training_data_devices'] = self.training_data_devices
            
            # 添加验证集性能指标（如果存在）
            if hasattr(self, 'validation_metrics') and self.validation_metrics:
                metadata['validation_metrics'] = self.validation_metrics
            
            # 添加多测试集性能（仅支持 test_set_performance）
            if hasattr(self, 'test_set_performance') and self.test_set_performance:
                metadata['test_set_performance'] = self.test_set_performance
            
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
            
        except Exception as e:
            print(f"预测失败: {e}")
            # 返回一个不确定的结果
            return ModelOutput(
                model_name=self.model_name,
                predictions={"0": 0.5, "1": 0.5},
                top_class="0",
                top_confidence=0.5,
                requires_mask=self.requires_mask,
                metadata={"error": str(e)}
            )
    
    def predict_batch(self, images: list, masks: list, show_progress: bool = True):
        """
        批量运行推理
        
        注意：
        - PyRadiomics 特征提取：不支持批量，内部逐张提取（但统一管理）
        - AutoGluon 预测：支持真正的批量预测（一次性处理所有样本）
        
        Args:
            images: 图像列表 [(H, W, C), ...]
            masks: 掩码列表 [(H, W), ...] - 必需
            show_progress: 是否显示进度
            
        Returns:
            ModelOutput 列表
        """
        if len(images) != len(masks):
            raise ValueError("图像和掩码数量必须匹配")
        
        # 批量提取特征（内部逐张处理，但统一管理）
        if show_progress:
            print(f"  批量提取 PyRadiomics 特征 ({len(images)} 张图像)...")
            print(f"  注意: PyRadiomics 不支持批量特征提取，将逐张处理")
        
        try:
            features_df = self.preprocess_batch(images, masks, show_progress=show_progress)
        except Exception as e:
            print(f"批量特征提取失败: {e}")
            # 回退到逐张处理
            if show_progress:
                print("  回退到逐张处理模式...")
            return [self.predict(img, msk) for img, msk in zip(images, masks)]
        
        # 批量预测（真正的批量，一次性处理所有样本）
        if show_progress:
            print(f"  批量预测 ({len(images)} 个样本，一次性处理)...", end=" ")
        
        try:
            # 获取预测概率（AutoGluon 支持批量预测）
            proba = self.predictor.predict_proba(features_df)
            
            if show_progress:
                print("✓")
            
            # 构建 ModelOutput 列表
            results = []
            
            for idx in range(len(images)):
                # 提取概率值
                if hasattr(proba, 'values'):
                    proba_array = proba.values[idx]
                elif hasattr(proba, 'iloc'):
                    proba_array = proba.iloc[idx].values
                else:
                    proba_array = proba[idx]
                
                # 构建预测字典
                if len(proba_array) == 2:
                    prob_class_0 = float(proba_array[0])
                    prob_class_1 = float(proba_array[1])
                else:
                    prob_class_1 = float(proba_array[0])
                    prob_class_0 = 1.0 - prob_class_1
                
                predictions = {
                    "0": prob_class_0,
                    "1": prob_class_1
                }
                
                # 找到最高置信度的类别
                top_class = max(predictions.items(), key=lambda x: x[1])
                
                # 获取最佳模型名称（如果可用）
                try:
                    best_model = self.predictor.model_best
                except:
                    best_model = "unknown"
                
                metadata = {
                    "framework": "AutoGluon + PyRadiomics",
                    "n_features": len(features_df.columns),
                    "best_model": best_model
                }

                # 添加多测试集性能（仅支持 test_set_performance）
                if hasattr(self, 'test_set_performance') and self.test_set_performance:
                    metadata['test_set_performance'] = self.test_set_performance
                
                # 添加训练数据设备信息（如果存在）
                if hasattr(self, 'training_data_devices') and self.training_data_devices:
                    metadata['training_data_devices'] = self.training_data_devices
                
                result = ModelOutput(
                    model_name=self.model_name,
                    predictions=predictions,
                    top_class=top_class[0],
                    top_confidence=top_class[1],
                    requires_mask=self.requires_mask,
                    metadata=metadata
                )
                results.append(result)
            
            if show_progress:
                print(f"  ✓ 批量预测完成，共处理 {len(results)} 个样本")
            
            return results
            
        except Exception as e:
            print(f"批量预测失败: {e}")
            # 回退到逐张处理
            if show_progress:
                print("  回退到逐张处理模式...")
            return [self.predict(img, msk) for img, msk in zip(images, masks)]

