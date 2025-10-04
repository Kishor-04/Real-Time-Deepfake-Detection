"""
Dataset and DataLoader creation for deepfake detection
Author: Kishor-04
Date: 2025-01-04
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        
        if image is None:
            # If image loading fails, return a blank image
            print(f"Warning: Could not load {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def prepare_data_splits(config_path='config.yaml'):
    """Prepare train/val/test splits from processed faces"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    faces_dir = Path(config['dataset']['processed_data_path']) / 'faces'
    
    if not faces_dir.exists():
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    # Real images (label = 0)
    real_dir = faces_dir / 'real'
    if real_dir.exists():
        for video_dir in real_dir.iterdir():
            if video_dir.is_dir():
                for img_path in video_dir.glob("*.jpg"):
                    image_paths.append(img_path)
                    labels.append(0)
    
    # Fake images (label = 1)
    fake_dir = faces_dir / 'fake'
    if fake_dir.exists():
        for video_dir in fake_dir.iterdir():
            if video_dir.is_dir():
                for img_path in video_dir.glob("*.jpg"):
                    image_paths.append(img_path)
                    labels.append(1)
    
    if len(image_paths) == 0:
        raise ValueError("No images found! Please run preprocessing first.")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Real images: {labels.count(0):,}")
    print(f"  Fake images: {labels.count(1):,}")
    print(f"  Total images: {len(labels):,}")
    
    # Split data
    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    
    # First split: train + (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=labels
    )
    
    # Second split: val + test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\n📋 Data Splits:")
    print(f"  Train: {len(X_train):,} images ({len(X_train)/len(labels)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} images ({len(X_val)/len(labels)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} images ({len(X_test)/len(labels)*100:.1f}%)")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

def create_data_loaders(config_path='config.yaml'):
    """Create DataLoaders for training"""
    import sys
    sys.path.append('.')
    from src.preprocessing.data_augmentation import get_train_transforms, get_val_transforms
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare splits
    data_splits = prepare_data_splits(config_path)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=get_train_transforms(config)
    )
    
    val_dataset = DeepfakeDataset(
        data_splits['val'][0],
        data_splits['val'][1],
        transform=get_val_transforms()
    )
    
    test_dataset = DeepfakeDataset(
        data_splits['test'][0],
        data_splits['test'][1],
        transform=get_val_transforms()
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n✅ DataLoaders created successfully")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader