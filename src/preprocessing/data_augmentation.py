"""
Data augmentation transforms for deepfake detection
Author: Kishor-04
Date: 2025-01-04
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(config):
    """Get training data augmentation transforms"""
    aug_config = config['augmentation']
    
    if aug_config['enable']:
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.Rotate(limit=aug_config['rotation_range'], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.3
            ),
            
            # Color transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Noise and blur (FIXED)
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            
            # Additional augmentations for robustness
            A.CoarseDropout(
                max_holes=8,
                max_height=8,
                max_width=8,
                min_holes=1,
                fill_value=0,
                p=0.2
            ),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # No augmentation, only normalization
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def get_val_transforms():
    """Get validation/test data transforms (no augmentation)"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])