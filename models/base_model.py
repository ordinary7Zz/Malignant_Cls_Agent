"""
Base model interface for classification models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelOutput:
    """
    Standard output format for model predictions
    """
    model_name: str
    predictions: Dict[str, float]  # class_name: confidence
    top_class: str
    top_confidence: float
    requires_mask: bool
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'model_name': self.model_name,
            'predictions': self.predictions,
            'top_class': self.top_class,
            'top_confidence': self.top_confidence,
            'requires_mask': self.requires_mask,
            'metadata': self.metadata or {}
        }


class BaseClassificationModel(ABC):
    """
    Abstract base class for all classification models
    """
    
    def __init__(self, model_name: str, model_path: str, requires_mask: bool = False):
        """
        Initialize the classification model
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to the model weights/checkpoint
            requires_mask: Whether this model requires a segmentation mask
        """
        self.model_name = model_name
        self.model_path = model_path
        self.requires_mask = requires_mask
        self.model = None
        self.class_names = []
        
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model from the specified path
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Any:
        """
        Preprocess the input image (and mask if required)
        
        Args:
            image: Input image as numpy array (H, W, C)
            mask: Optional segmentation mask as numpy array (H, W)
            
        Returns:
            Preprocessed input ready for model inference
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> ModelOutput:
        """
        Run inference on the input image
        
        Args:
            image: Input image as numpy array (H, W, C)
            mask: Optional segmentation mask as numpy array (H, W)
            
        Returns:
            ModelOutput containing predictions and metadata
        """
        pass
    
    def validate_inputs(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """
        Validate input images and masks
        
        Args:
            image: Input image
            mask: Optional mask
            
        Raises:
            ValueError: If inputs are invalid
        """
        if image is None:
            raise ValueError("Input image cannot be None")
        
        if self.requires_mask and mask is None:
            raise ValueError(f"Model {self.model_name} requires a segmentation mask")
        
        if mask is not None and image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image shape {image.shape[:2]} does not match mask shape {mask.shape[:2]}"
            )
    
    def predict_batch(self, images: list, masks: list = None, show_progress: bool = True) -> List[ModelOutput]:
        """
        Run inference on multiple images (default implementation: sequential)
        Subclasses can override this for true batch processing
        
        Args:
            images: List of input images as numpy arrays [(H, W, C), ...]
            masks: Optional list of segmentation masks [(H, W), ...]
            show_progress: Whether to show progress
            
        Returns:
            List of ModelOutput containing predictions
        """
        if masks is None:
            masks = [None] * len(images)
        
        if len(images) != len(masks):
            raise ValueError("Number of images and masks must match")
        
        results = []
        for idx, (image, mask) in enumerate(zip(images, masks)):
            if show_progress:
                print(f"    [{idx+1}/{len(images)}]...", end=" ")
            
            try:
                result = self.predict(image, mask)
                results.append(result)
                
                if show_progress:
                    print(f"✓ {result.top_class} ({result.top_confidence:.4f})")
            except Exception as e:
                if show_progress:
                    print(f"✗ 失败: {e}")
                # 添加一个失败的占位结果
                results.append(None)
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'name': self.model_name,
            'path': self.model_path,
            'requires_mask': self.requires_mask,
            'class_names': self.class_names
        }

