"""
Image processing utilities
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union
from pathlib import Path


class ImageProcessor:
    """
    Utility class for image and mask processing
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize image processor
        
        Args:
            target_size: Target image size (height, width)
            normalize_mean: Mean values for normalization (R, G, B)
            normalize_std: Std values for normalization (R, G, B)
        """
        self.target_size = target_size
        self.normalize_mean = np.array(normalize_mean).reshape(1, 1, 3)
        self.normalize_std = np.array(normalize_std).reshape(1, 1, 3)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL (handles various formats)
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        return image_np
    
    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """
        Load segmentation mask from file
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Mask as numpy array (H, W)
        """
        mask_path = Path(mask_path)
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Load mask (typically grayscale)
        mask = Image.open(mask_path)
        
        # Convert to grayscale if needed
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Convert to numpy array
        mask_np = np.array(mask)
        
        return mask_np
    
    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image (H, W, C)
            target_size: Target size (height, width), uses default if None
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        # PIL resize expects (width, height)
        target_size_wh = (target_size[1], target_size[0])
        
        image_pil = Image.fromarray(image)
        image_resized = image_pil.resize(target_size_wh, Image.BILINEAR)
        
        return np.array(image_resized)
    
    def resize_mask(
        self,
        mask: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize mask to target size using nearest neighbor
        
        Args:
            mask: Input mask (H, W)
            target_size: Target size (height, width), uses default if None
            
        Returns:
            Resized mask
        """
        if target_size is None:
            target_size = self.target_size
        
        # PIL resize expects (width, height)
        target_size_wh = (target_size[1], target_size[0])
        
        mask_pil = Image.fromarray(mask)
        mask_resized = mask_pil.resize(target_size_wh, Image.NEAREST)
        
        return np.array(mask_resized)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics
        
        Args:
            image: Input image (H, W, C) with values in [0, 255]
            
        Returns:
            Normalized image with values roughly in [-1, 1]
        """
        # Convert to float and scale to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Normalize
        image_normalized = (image_float - self.normalize_mean) / self.normalize_std
        
        return image_normalized
    
    def apply_mask_to_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_value: int = 0
    ) -> np.ndarray:
        """
        Apply mask to image (keep only masked region)
        
        Args:
            image: Input image (H, W, C)
            mask: Binary mask (H, W), non-zero values indicate regions to keep
            background_value: Value for background (masked-out) pixels
            
        Returns:
            Masked image
        """
        # Ensure mask is binary
        mask_binary = (mask > 0).astype(np.uint8)
        
        # Expand mask to match image channels
        mask_3ch = np.expand_dims(mask_binary, axis=2)
        mask_3ch = np.repeat(mask_3ch, 3, axis=2)
        
        # Apply mask
        masked_image = image * mask_3ch + background_value * (1 - mask_3ch)
        
        return masked_image.astype(image.dtype)
    
    def preprocess(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        resize: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (H, W, C)
            mask: Optional mask (H, W)
            resize: Whether to resize to target size
            normalize: Whether to normalize
            
        Returns:
            Tuple of (processed_image, processed_mask)
        """
        processed_image = image.copy()
        processed_mask = mask.copy() if mask is not None else None
        
        # Resize
        if resize:
            processed_image = self.resize_image(processed_image)
            if processed_mask is not None:
                processed_mask = self.resize_mask(processed_mask)
        
        # Normalize
        if normalize:
            processed_image = self.normalize_image(processed_image)
        
        return processed_image, processed_mask
    
    @staticmethod
    def visualize_mask_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Create visualization with mask overlay
        
        Args:
            image: Input image (H, W, C)
            mask: Binary mask (H, W)
            alpha: Transparency of overlay
            color: Color for mask overlay (R, G, B)
            
        Returns:
            Image with mask overlay
        """
        # Create colored overlay
        overlay = np.zeros_like(image)
        mask_binary = (mask > 0)
        overlay[mask_binary] = color
        
        # Blend
        visualization = cv2.addWeighted(
            image.astype(np.uint8),
            1 - alpha,
            overlay.astype(np.uint8),
            alpha,
            0
        )
        
        return visualization
    
    @staticmethod
    def save_image(image: np.ndarray, save_path: Union[str, Path]) -> None:
        """
        Save image to file
        
        Args:
            image: Image array
            save_path: Path to save image
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_pil.save(save_path)

