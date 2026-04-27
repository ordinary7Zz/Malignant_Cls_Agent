"""
Model registry for managing multiple classification models
"""

from typing import Any, Dict, List, Optional

import numpy as np

from calibration.runtime import maybe_apply_calibration_map

from .base_model import BaseClassificationModel, ModelOutput


class ModelRegistry:
    """
    Registry to manage and coordinate multiple classification models
    """
    
    def __init__(self):
        """Initialize the model registry"""
        self.models: Dict[str, BaseClassificationModel] = {}
        self._loaded = False
        # model_name -> 离线校准 JSON artifact；为 None 或未设置时不做校准
        self.calibration_map: Optional[Dict[str, Dict[str, Any]]] = None
    
    def register_model(self, model: BaseClassificationModel) -> None:
        """
        Register a new model
        
        Args:
            model: Instance of BaseClassificationModel
        """
        if model.model_name in self.models:
            print(f"Warning: Model {model.model_name} already registered. Overwriting.")
        
        self.models[model.model_name] = model
        print(f"Registered model: {model.model_name}")
    
    def unregister_model(self, model_name: str) -> None:
        """
        Remove a model from the registry
        
        Args:
            model_name: Name of the model to remove
        """
        if model_name in self.models:
            del self.models[model_name]
            print(f"Unregistered model: {model_name}")
        else:
            print(f"Model {model_name} not found in registry")
    
    def load_all_models(self) -> None:
        """Load all registered models"""
        print("Loading all models...")
        for name, model in self.models.items():
            try:
                model.load_model()
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {str(e)}")
        self._loaded = True
    
    def predict_all(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> List[ModelOutput]:
        """
        Run inference with all registered models
        
        Args:
            image: Input image as numpy array (H, W, C)
            mask: Optional segmentation mask as numpy array (H, W)
            
        Returns:
            List of ModelOutput from all models
        """
        if not self._loaded:
            self.load_all_models()
        
        results = []
        
        for name, model in self.models.items():
            try:
                # Validate inputs for this model
                model.validate_inputs(image, mask)
                
                # Run prediction
                output = model.predict(image, mask)
                try:
                    maybe_apply_calibration_map(output, self.calibration_map)
                except Exception:
                    pass
                results.append(output)

                print(f"✓ {name}: {output.top_class} ({output.top_confidence:.4f})")
                
            except Exception as e:
                print(f"✗ Error with {name}: {str(e)}")
        
        return results
    
    def get_model(self, model_name: str) -> Optional[BaseClassificationModel]:
        """
        Get a specific model by name
        
        Args:
            model_name: Name of the model
            
        Returns:
            The model instance or None if not found
        """
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        Get list of all registered model names
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_models_info(self) -> List[Dict]:
        """
        Get information about all registered models
        
        Returns:
            List of model information dictionaries
        """
        return [model.get_info() for model in self.models.values()]
    
    def __len__(self) -> int:
        """Return number of registered models"""
        return len(self.models)
    
    def __repr__(self) -> str:
        """String representation of the registry"""
        return f"ModelRegistry(models={len(self.models)}, loaded={self._loaded})"

