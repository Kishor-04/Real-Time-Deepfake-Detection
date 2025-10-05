"""
EfficientNet-based deepfake detection model
Author: Kishor-04
Date: 2025-01-04
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import os

class EfficientNetDeepfakeDetector(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, model_name='efficientnet-b0', num_classes=2, pretrained=True):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        # Load EfficientNet with ImageNet weights (built-in)
        if pretrained:
            print(f"  Loading {model_name} with ImageNet pre-trained weights...")
            self.model = EfficientNet.from_pretrained(model_name)
        else:
            self.model = EfficientNet.from_name(model_name)
        
        # Modify the final classification layer
        in_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"  ✓ {model_name} loaded successfully")
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for name, param in self.model.named_parameters():
            if '_fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True

def load_pretrained_efficientnet(model_name='efficientnet-b0', weights_path=None, num_classes=2):
    """Load EfficientNet model with optional deepfake pre-trained weights"""
    
    if weights_path and os.path.exists(weights_path):
        print(f"  Loading deepfake-specific weights from {weights_path}")
        
        # Load model without pretrained weights first
        model = EfficientNetDeepfakeDetector(
            model_name=model_name, 
            num_classes=num_classes, 
            pretrained=False
        )
        
        # FIXED: Load deepfake-specific weights with proper error handling
        try:
            # Try with weights_only=True first (safer)
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        except Exception as e:
            # Fallback to weights_only=False if the file contains non-tensor objects
            print(f"  ⚠️  Using weights_only=False due to older checkpoint format")
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                # Try loading directly
                try:
                    model.load_state_dict(checkpoint, strict=False)
                except Exception as load_error:
                    print(f"  ⚠️  Warning: Could not load checkpoint format: {load_error}")
                    print(f"  ⚠️  Falling back to ImageNet weights instead.")
                    return EfficientNetDeepfakeDetector(
                        model_name=model_name, 
                        num_classes=num_classes, 
                        pretrained=True
                    )
        else:
            try:
                model.load_state_dict(checkpoint, strict=False)
            except Exception as load_error:
                print(f"  ⚠️  Warning: Could not load checkpoint: {load_error}")
                print(f"  ⚠️  Falling back to ImageNet weights instead.")
                return EfficientNetDeepfakeDetector(
                    model_name=model_name, 
                    num_classes=num_classes, 
                    pretrained=True
                )
        
        print("  ✓ Deepfake-specific weights loaded successfully")
    else:
        # Use ImageNet pre-trained weights (built-in)
        if weights_path:
            print(f"  ⚠️  Weights file not found: {weights_path}")
        print("  Using ImageNet pre-trained weights (built-in)")
        model = EfficientNetDeepfakeDetector(
            model_name=model_name, 
            num_classes=num_classes, 
            pretrained=True
        )
    
    return model